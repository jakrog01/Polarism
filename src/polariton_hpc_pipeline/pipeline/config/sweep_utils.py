"""Shared utilities for scenario sweep expansion.

This module contains geometry, naming, timing and power-resolution helpers used
by both the generic sweep implementation and experiment-specific sweep modes.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from pipeline.config.loader import resolve_delay, resolve_power

PULSE_LASER_TYPES: frozenset[str] = frozenset({"pulse-gaussian"})

SQUARE4_CORNERS: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, 1.0),
)

P_RELATIVE_POWER_RE = re.compile(
    r"^\s*([0-9]*\.?[0-9]*)\s*"
    r"(P(?:_(?:ring|probe|assisted|assisted_probe|probe_assisted|"
    r"assisted_ring|ring_assisted))?"
    r"|P(?:ring|probe|assisted|assistedprobe|probeassisted|"
    r"assistedring|ringassisted))\s*$",
    re.IGNORECASE,
)


def apply_square_side(scenario: dict[str, Any], a: float) -> None:
    """Place the first four lasers at square corners with side length ``a``."""
    half = a / 2.0
    for i, ldef in enumerate(scenario.get("lasers", [])[:4]):
        cx, cy = SQUARE4_CORNERS[i]
        ldef["x0"] = cx * half
        ldef["y0"] = cy * half


def resolve_laser_type(ldef: dict[str, Any], defaults: dict[str, Any]) -> str:
    """Return the effective laser_type for a laser definition."""
    return str(ldef.get("laser_type", defaults.get("laser_type", "pulse-gaussian")))


def fmt_value(value: float) -> str:
    """Return a filename-safe compact float label."""
    text = f"{float(value):.6g}"
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def fmt_label(value: Any) -> str:
    """Return a filename-safe label for numeric or symbolic sweep values."""
    text = str(value)
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def power_suffix_prefix(power_definition: str) -> str:
    """Return the scenario-name prefix for the swept pump-strength value."""
    return "E" if power_definition == "pulse_energy" else "P"


def power_reference_expr(coefficient: float, symbol: str = "P") -> str:
    """Return a compact P-relative expression such as ``0.75P_ring``."""
    coeff = f"{float(coefficient):.6g}"
    return f"{coeff}{symbol}"


def is_reference_power(expr: Any) -> bool:
    """Return whether *expr* depends on the sweep reference power ``P``."""
    if expr is None:
        return True
    if isinstance(expr, str):
        return P_RELATIVE_POWER_RE.match(expr) is not None
    return False


def selected_base_scenarios(
    scenarios: list[dict[str, Any]],
    sweep_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return scenarios selected by ``parameter_sweep.base_scenarios``."""
    selected = sweep_cfg.get("base_scenarios")
    if not selected:
        return scenarios
    selected_set = {str(name) for name in selected}
    return [sc for sc in scenarios if sc.get("name") in selected_set]


def apply_absolute_power(
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


def uses_absolute_power_label(scenario: dict[str, Any]) -> bool:
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
        if not is_reference_power(power_expr):
            return True
    return False


def representative_resolved_power(scenario: dict[str, Any], fallback: float) -> float:
    """Return a compact power label for an already resolved scenario."""
    return max(
        (float(ld.get("power", fallback)) for ld in scenario.get("lasers", [{}])),
        default=float(fallback),
    )


def apply_timing_grid_point(
    scenario: dict[str, Any],
    pulse_separation: float,
    sigma_time: float,
    cutoff_sigma: float,
) -> None:
    """Inject one temporal grid point into a copied scenario."""
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
