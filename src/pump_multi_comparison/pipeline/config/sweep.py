"""Parameter-sweep expansion for pump_multi_comparison runs."""
from __future__ import annotations

import copy
import re
from typing import Any

from pipeline.config.loader import resolve_delay, resolve_power

_PULSE_LASER_TYPES: frozenset[str] = frozenset({"pulse-gaussian"})

_SQUARE4_CORNERS: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    ( 1.0, -1.0),
    (-1.0,  1.0),
    ( 1.0,  1.0),
)


def _apply_square_side(scenario: dict[str, Any], a: float) -> None:
    half = a / 2.0
    for i, ldef in enumerate(scenario.get("lasers", [])[:4]):
        cx, cy = _SQUARE4_CORNERS[i]
        ldef["x0"] = cx * half
        ldef["y0"] = cy * half
_P_RELATIVE_POWER_RE = re.compile(
    r"^\s*([0-9]*\.?[0-9]*)\s*"
    r"(P(?:_(?:ring|probe|assisted|assisted_probe|probe_assisted|"
    r"assisted_ring|ring_assisted))?"
    r"|P(?:ring|probe|assisted|assistedprobe|probeassisted|"
    r"assistedring|ringassisted))\s*$",
    re.IGNORECASE,
)


def parameter_sweep_enabled(cfg: dict[str, Any]) -> bool:
    """Return whether the config requests direct parameter-sweep mode."""
    return bool(cfg.get("global", {}).get("parameter_sweep", {}).get("enabled", False))


def resolve_laser_type(ldef: dict[str, Any], defaults: dict[str, Any]) -> str:
    """Return the effective laser_type for a laser definition."""
    return str(ldef.get("laser_type", defaults.get("laser_type", "pulse-gaussian")))


def _fmt_value(value: float) -> str:
    text = f"{float(value):.6g}"
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _power_suffix_prefix(power_definition: str) -> str:
    """Return the scenario-name prefix for the swept pump-strength value."""
    return "E" if power_definition == "pulse_energy" else "P"


def _is_reference_power(expr: Any) -> bool:
    """Return whether *expr* depends on the sweep reference power ``P``."""
    if expr is None:
        return True
    if isinstance(expr, str):
        return _P_RELATIVE_POWER_RE.match(expr) is not None
    return False


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


def _uses_absolute_power_label(scenario: dict[str, Any]) -> bool:
    """Return whether scenario naming should use resolved absolute laser power."""
    laser_defs: list[dict[str, Any]] = scenario.get("lasers", [])
    modifiers: list[dict[str, Any]] = scenario.get("power_modifiers", [])
    ids = [str(ldef.get("id", f"laser_{i}")) for i, ldef in enumerate(laser_defs)]

    for i, ldef in enumerate(laser_defs):
        tags = ldef.get("tags") or []
        power_expr = ldef.get("power")
        for mod in modifiers:
            if ids[i] in mod.get("ids", []) or any(t in tags for t in mod.get("tags", [])):
                power_expr = mod.get("power")
        if not _is_reference_power(power_expr):
            return True
    return False


def _representative_resolved_power(scenario: dict[str, Any], fallback: float) -> float:
    """Return a compact power label for an already resolved scenario."""
    return max(
        (float(ld.get("power", fallback)) for ld in scenario.get("lasers", [{}])),
        default=float(fallback),
    )


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


def probe5_sweep_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when the config requests probe5 mode."""
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    return bool(ps.get("enabled", False)) and str(ps.get("mode", "")) == "probe5"


def probe5_ring_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should run a ring-only calibration first."""
    if not probe5_sweep_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    ring_cal = ps.get("ring_calibration", {})
    return bool(ring_cal.get("enabled", False))


def probe5_probe_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should also calibrate the center probe alone."""
    if not probe5_ring_calibration_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    probe_cal = ps.get("probe_calibration", {})
    return bool(probe_cal.get("enabled", False))


def probe5_assisted_probe_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should calibrate the probe inside a ring trap."""
    if not probe5_ring_calibration_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    assisted_cal = ps.get("assisted_probe_calibration", {})
    return bool(assisted_cal.get("enabled", False))


def probe5_assisted_ring_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should calibrate ring strength for a fixed probe."""
    if not probe5_ring_calibration_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    assisted_cal = ps.get("assisted_ring_calibration", {})
    return bool(assisted_cal.get("enabled", False))


def _fmt_label(value: Any) -> str:
    text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _p_expr(coefficient: float, symbol: str = "P") -> str:
    coeff = f"{float(coefficient):.6g}"
    return f"{coeff}{symbol}"


def _apply_probe5_sweep_point(
    scenario: dict[str, Any],
    probe_laser_id: str,
    ring_energy: Any,
    probe_energy: Any,
    probe_delay: float,
) -> None:
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = ring_energy

    found_probe = False
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = probe_energy
            ldef["delay"] = float(probe_delay)
            found_probe = True
            break
    if not found_probe:
        raise ValueError(f"probe5 laser id {probe_laser_id!r} not found")


def _probe5_ring_energy_axis(
    sweep_cfg: dict[str, Any],
) -> list[tuple[Any, float]]:
    ring_cal = sweep_cfg.get("ring_calibration", {})
    if bool(ring_cal.get("enabled", False)):
        if "ring_multiplier_values" in sweep_cfg:
            reference = str(sweep_cfg.get("ring_multiplier_reference", "P_ring"))
            return [
                (_p_expr(float(multiplier), reference), float(multiplier))
                for multiplier in sweep_cfg.get("ring_multiplier_values", [1.0])
            ]
        safe_fraction = float(ring_cal.get("safe_fraction", 0.75))
        return [
            (_p_expr(safe_fraction * float(scale), "P_ring"), float(scale))
            for scale in sweep_cfg.get("ring_scale_values", [1.0])
        ]

    ring_energy_ref = float(sweep_cfg["ring_energy"])

    if "ring_energy_values" in sweep_cfg:
        return [
            (float(value), float(value) / ring_energy_ref)
            for value in sweep_cfg["ring_energy_values"]
        ]

    if "ring_scale_values" in sweep_cfg:
        return [
            (ring_energy_ref * float(scale), float(scale))
            for scale in sweep_cfg["ring_scale_values"]
        ]

    return [(ring_energy_ref, 1.0)]


def expand_probe5_sweep(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Expand a probe5 scenario over independent probe-delay/energy axes.

    The four corner ring lasers sit at ``square_side`` and can be swept through
    ``ring_scale_values`` or ``ring_energy_values``.  The fifth laser
    (identified by ``probe_laser_id``) is the delayed center probe.

    Returns
    -------
    (expanded_cfg, names, threshold_stub)
        Expanded config snapshot, list of scenario names, and a synthetic
        threshold stub that lets the GPU stage skip threshold search.
    """
    expanded_cfg = copy.deepcopy(cfg)
    global_cfg = expanded_cfg["global"]
    sweep_cfg = global_cfg["parameter_sweep"]
    defaults = global_cfg.get("laser_defaults", {})
    power_definition = str(defaults.get("power_definition", "peak_amplitude"))
    ring_calibrated = bool((sweep_cfg.get("ring_calibration") or {}).get("enabled", False))

    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))
    square_side = float(sweep_cfg["square_side"])
    ring_energy_ref = float(sweep_cfg.get("ring_energy", 1.0))
    ring_energy_axis = _probe5_ring_energy_axis(sweep_cfg)
    ring_axis_is_swept = (
        "ring_scale_values" in sweep_cfg
        or "ring_energy_values" in sweep_cfg
        or "ring_multiplier_values" in sweep_cfg
    )
    probe_delay_values = [float(v) for v in sweep_cfg["probe_delay_values"]]
    probe_energy_values = list(sweep_cfg["probe_energy_values"])
    sigma_time = float(defaults.get("sigma_time", 1.5))
    cutoff_sigma = float(defaults.get("cutoff_sigma", 3.0))
    sigma_space = float(defaults.get("sigma_space", 2.0))
    pulse_sep = float(sweep_cfg.get("pulse_separation", 10.0))

    base_scenarios = _selected_base_scenarios(expanded_cfg.get("scenarios", []), sweep_cfg)
    expanded_scenarios: list[dict[str, Any]] = []

    for base in base_scenarios:
        base_name = str(base["name"])
        first_laser = (base.get("lasers") or [{}])[0]
        n_ring_pulses = int(first_laser.get("n_pulses", 9))
        pump_end_ps = (n_ring_pulses - 1) * pulse_sep + 2.0 * cutoff_sigma * sigma_time

        for probe_delay in probe_delay_values:
            for ring_energy, ring_scale in ring_energy_axis:
                for probe_energy in probe_energy_values:
                    sc = copy.deepcopy(base)
                    _apply_square_side(sc, square_side)
                    _apply_timing_grid_point(sc, pulse_sep, sigma_time, cutoff_sigma)
                    _apply_probe5_sweep_point(
                        sc, probe_laser_id, ring_energy, probe_energy, probe_delay
                    )
                    for ldef in sc.get("lasers", []):
                        ldef["sigma_space"] = float(sigma_space)

                    a_str = _fmt_value(square_side)
                    ring_scale_str = _fmt_value(ring_scale)
                    ering_str = _fmt_label(ring_energy)
                    eprobe_str = _fmt_label(probe_energy)
                    d_str = _fmt_value(probe_delay)
                    ring_part = (
                        f"_Rscale{ring_scale_str}_Ering{ering_str}"
                        if ring_axis_is_swept
                        else f"_Ering{ering_str}"
                    )
                    sc["name"] = (
                        f"{base_name}_a{a_str}{ring_part}"
                        f"_Eprobe{eprobe_str}_d{d_str}"
                    )

                    sc["sweep"] = {
                        "base_scenario": base_name,
                        "mode": "probe5",
                        "square_side": square_side,
                        "ring_energy": ring_energy,
                        "ring_energy_reference": ring_energy_ref,
                        "ring_scale": ring_scale,
                        "probe_energy": probe_energy,
                        "probe_delay_ps": probe_delay,
                        "pump_end_ps": pump_end_ps,
                        "sigma_time": sigma_time,
                        "sigma_space": sigma_space,
                        "n_ring_pulses": n_ring_pulses,
                        "probe_laser_id": probe_laser_id,
                        "power_definition": power_definition,
                        "power_label": _power_suffix_prefix(power_definition),
                    }
                    expanded_scenarios.append(sc)

    expanded_cfg["scenarios"] = expanded_scenarios
    names = [sc["name"] for sc in expanded_scenarios]

    grid = global_cfg.get("grid", {})
    threshold_stub: dict[str, Any] = {
        "search_completed": True,
        "mode": "parameter_sweep",
        "P_threshold": 1.0,
        "power_definition": power_definition,
        "sigma_time": sigma_time,
        "cutoff_sigma": cutoff_sigma,
        "sigma_space": sigma_space,
        "pulse_separation": pulse_sep,
        "lx": float(grid.get("lx", 160.0)),
        "ly": float(grid.get("ly", 160.0)),
    }
    if ring_calibrated:
        threshold_stub = None
    return expanded_cfg, names, threshold_stub


def expand_parameter_sweep(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Expand base scenarios over power (and optionally pulse-separation) values.

    Returns an expanded config snapshot, expanded scenario names, and a synthetic
    threshold result that lets the existing scenario stage run without invoking
    threshold search.

    For pulse-gaussian laser types, the expansion includes the pulse_separation
    axis and produces scenario names with a ``_sep{value}`` component.  For all
    other laser types (continuous-gaussian, etc.) the pulse_separation axis is
    omitted — ``pulse_separation_values`` is not required in the config, and
    scenario names carry only the power (and optionally sigma_space) component.

    For scenarios with ``geometry: square4``, the ``square_side_values`` axis is
    expanded instead of the pulse-separation axis.  Four laser positions are set
    to the corners ``(±a/2, ±a/2)`` for each side length ``a``, and the scenario
    name carries an ``_a{value}`` component.  The first pulse-separation value is
    still applied as fixed laser timing and recorded in metadata, but no
    ``_sep`` suffix is added; other scenario types retain their existing
    separation-axis behaviour regardless of ``n_pulses``.
    """
    if probe5_sweep_enabled(cfg):
        return expand_probe5_sweep(cfg)

    if not parameter_sweep_enabled(cfg):
        names = [sc["name"] for sc in cfg.get("scenarios", [])]
        return copy.deepcopy(cfg), names, {}

    expanded_cfg = copy.deepcopy(cfg)
    global_cfg = expanded_cfg["global"]
    sweep_cfg = global_cfg["parameter_sweep"]
    defaults = global_cfg.get("laser_defaults", {})
    threshold_cfg = global_cfg.get("threshold_search", {})
    power_definition = str(defaults.get("power_definition", "peak_amplitude"))
    power_prefix = _power_suffix_prefix(power_definition)

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

    base_scenarios = _selected_base_scenarios(expanded_cfg.get("scenarios", []), sweep_cfg)
    expanded_scenarios: list[dict[str, Any]] = []

    multiple_sigma = len(sigma_times) > 1
    multiple_sigma_space = len(sigma_spaces) > 1
    any_pulse_scenario = False

    for base in base_scenarios:
        base_name = str(base["name"])
        is_square4 = base.get("geometry") == "square4"
        scenario_has_pulse = any(
            resolve_laser_type(ldef, defaults) in _PULSE_LASER_TYPES
            for ldef in base.get("lasers", [])
        )
        if scenario_has_pulse:
            any_pulse_scenario = True
        use_resolved_power_label = _uses_absolute_power_label(base)
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
                            _apply_absolute_power(sc, power)
                            sweep_power = (
                                _representative_resolved_power(sc, power)
                                if use_resolved_power_label
                                else float(power)
                            )
                            side_prefix = (
                                f"a{_fmt_value(square_side)}_"
                                if is_square4 and square_side is not None
                                else ""
                            )
                            suffix = f"{side_prefix}{power_prefix}{_fmt_value(sweep_power)}"
                            if scenario_has_pulse and pulse_sep is not None and not is_square4:
                                suffix += f"_sep{_fmt_value(pulse_sep)}"
                            if multiple_sigma:
                                suffix += f"_sig{_fmt_value(sigma_time)}"
                            if multiple_sigma_space:
                                suffix += f"_sp{_fmt_value(sigma_space)}"
                            sc["name"] = f"{base_name}_{suffix}"
                            if scenario_has_pulse and pulse_sep is not None:
                                _apply_timing_grid_point(sc, pulse_sep, sigma_time, cutoff_sigma)
                            if is_square4 and square_side is not None:
                                _apply_square_side(sc, square_side)
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
