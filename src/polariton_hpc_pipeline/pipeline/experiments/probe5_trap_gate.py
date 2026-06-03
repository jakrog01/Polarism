"""Probe5 trap-gate experiment: 4 fixed ring spots + swept central probe.

Owns all probe5-specific pipeline logic:
  - mode detection and config validation
  - scenario expansion over (probe_delay, ring_multiplier, probe_energy) axes
  - calibration scenario builders (ring-only, probe-only, assisted variants)
  - threshold summary CSV/JSON and 2-D heatmaps
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
from typing import Any

import numpy as np

from pipeline.config.loader import resolve_delay, resolve_power
from pipeline.config.sweep import probe5_sweep_enabled
from pipeline.config.sweep_utils import (
    apply_square_side,
    apply_timing_grid_point,
    fmt_label,
    fmt_value,
    power_reference_expr,
    power_suffix_prefix,
    selected_base_scenarios,
)

_PROBE_POWER_REF_RE = re.compile(
    r"^\s*[0-9]*\.?[0-9]*\s*P_?probe\s*$",
    re.IGNORECASE,
)
_ASSISTED_PROBE_POWER_REF_RE = re.compile(
    r"^\s*[0-9]*\.?[0-9]*\s*P_?(?:assisted|assisted_?probe|probe_?assisted)\s*$",
    re.IGNORECASE,
)
_ASSISTED_RING_POWER_REF_RE = re.compile(
    r"^\s*[0-9]*\.?[0-9]*\s*P_?(?:assisted_?ring|ring_?assisted)\s*$",
    re.IGNORECASE,
)

_PROBE5_THRESHOLD_CRITERION: float = 0.05

_PROBE5_CSV_COLUMNS: list[str] = [
    "scenario", "probe_delay_ps", "probe_energy", "ring_energy", "ring_scale",
    "square_side_um", "sigma_space", "pump_end_ps",
    "central_roi_peak_psi_sq", "central_roi_peak_emission",
    "central_roi_peak_nR", "t_peak_emission_ps",
    "crossed_threshold",
]


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
                (power_reference_expr(float(multiplier), reference), float(multiplier))
                for multiplier in sweep_cfg.get("ring_multiplier_values", [1.0])
            ]
        safe_fraction = float(ring_cal.get("safe_fraction", 0.75))
        return [
            (power_reference_expr(safe_fraction * float(scale), "P_ring"), float(scale))
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


def _expand_probe5_sweep(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Expand a probe5 scenario over independent probe-delay/energy axes."""
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

    base_scenarios = selected_base_scenarios(expanded_cfg.get("scenarios", []), sweep_cfg)
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
                    apply_square_side(sc, square_side)
                    apply_timing_grid_point(sc, pulse_sep, sigma_time, cutoff_sigma)
                    _apply_probe5_sweep_point(
                        sc, probe_laser_id, ring_energy, probe_energy, probe_delay
                    )
                    for ldef in sc.get("lasers", []):
                        ldef["sigma_space"] = float(sigma_space)

                    a_str = fmt_value(square_side)
                    ring_scale_str = fmt_value(ring_scale)
                    ering_str = fmt_label(ring_energy)
                    eprobe_str = fmt_label(probe_energy)
                    d_str = fmt_value(probe_delay)
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
                        "power_label": power_suffix_prefix(power_definition),
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


def _probe5_ring_calibration_scenario(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build the strict four-spot ring-only calibration scenario."""
    if not probe5_ring_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = None
    for sc in candidates:
        sweep = sc.get("sweep") or {}
        try:
            if (
                float(sweep.get("ring_scale", -1.0)) == 1.0
                and float(sweep.get("probe_energy", -1.0)) == 0.0
            ):
                source = sc
                break
        except (TypeError, ValueError):
            continue
    if source is None and candidates:
        source = candidates[0]
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_ring_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = "P"
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = 0.0
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_ring_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
    }
    return scenario


def _probe5_probe_calibration_scenario(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build the strict center-probe-only calibration scenario."""
    if not probe5_probe_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = None
    for sc in candidates:
        sweep = sc.get("sweep") or {}
        try:
            if float(sweep.get("ring_scale", -1.0)) == 0.0:
                source = sc
                break
        except (TypeError, ValueError):
            continue
    if source is None and candidates:
        source = candidates[0]
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_probe_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = 0.0
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = "P"
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_probe_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
    }
    return scenario


def _probe5_assisted_probe_calibration_scenario(
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Build center-probe calibration inside a fixed subthreshold ring trap."""
    if not probe5_assisted_probe_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))
    ring_cal = sweep_cfg.get("ring_calibration") or {}
    assisted_cal = sweep_cfg.get("assisted_probe_calibration") or {}
    ring_fraction = float(
        assisted_cal.get("ring_fraction", ring_cal.get("safe_fraction", 0.75))
    )

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = None
    for sc in candidates:
        sweep = sc.get("sweep") or {}
        try:
            if float(sweep.get("ring_scale", -1.0)) == 1.0:
                source = sc
                break
        except (TypeError, ValueError):
            continue
    if source is None and candidates:
        source = candidates[0]
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_assisted_probe_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = f"{ring_fraction:.6g}P_ring"
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = "P"
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_assisted_probe_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
        "ring_fraction": ring_fraction,
    }
    return scenario


def _probe5_assisted_ring_calibration_scenario(
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Build ring-strength calibration with a fixed subthreshold center probe."""
    if not probe5_assisted_ring_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))
    assisted_cal = sweep_cfg.get("assisted_ring_calibration") or {}
    probe_fraction = float(assisted_cal.get("probe_fraction", 0.75))

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = candidates[0] if candidates else None
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_assisted_ring_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = "P"
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = f"{probe_fraction:.6g}P_probe"
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_assisted_ring_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
        "probe_fraction": probe_fraction,
    }
    return scenario


def build_probe5_calibration_scenarios(
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any] | None]:
    """Return all four calibration scenario dicts (None when phase is disabled)."""
    return {
        "ring": _probe5_ring_calibration_scenario(cfg),
        "probe": _probe5_probe_calibration_scenario(cfg),
        "assisted_probe": _probe5_assisted_probe_calibration_scenario(cfg),
        "assisted_ring": _probe5_assisted_ring_calibration_scenario(cfg),
    }


def _valid_power_expr(expr: Any) -> bool:
    if expr is None:
        return False
    try:
        resolve_power(expr, 1.0)
        return True
    except (ValueError, TypeError):
        return False


def _validate_probe5_sweep(
    sweep_cfg: dict[str, Any],
    defaults: dict[str, Any],
    cfg: dict[str, Any],
) -> list[str]:
    """Validate the probe5-specific sweep section."""
    errors: list[str] = []

    probe_delay_values = sweep_cfg.get("probe_delay_values")
    probe_energy_values = sweep_cfg.get("probe_energy_values")
    ring_scale_values = sweep_cfg.get("ring_scale_values")
    ring_multiplier_values = sweep_cfg.get("ring_multiplier_values")
    ring_energy_values = sweep_cfg.get("ring_energy_values")
    ring_calibration = sweep_cfg.get("ring_calibration") or {}
    ring_calibration_enabled = bool(ring_calibration.get("enabled", False))
    probe_calibration = sweep_cfg.get("probe_calibration") or {}
    probe_calibration_enabled = bool(probe_calibration.get("enabled", False))
    assisted_calibration = sweep_cfg.get("assisted_probe_calibration") or {}
    assisted_calibration_enabled = bool(assisted_calibration.get("enabled", False))
    assisted_ring_calibration = sweep_cfg.get("assisted_ring_calibration") or {}
    assisted_ring_calibration_enabled = bool(
        assisted_ring_calibration.get("enabled", False)
    )
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))

    if "enabled" in probe_calibration and not isinstance(probe_calibration.get("enabled"), bool):
        errors.append("parameter_sweep.probe_calibration.enabled must be boolean")
    if "enabled" in assisted_calibration and not isinstance(
        assisted_calibration.get("enabled"), bool
    ):
        errors.append("parameter_sweep.assisted_probe_calibration.enabled must be boolean")
    if "enabled" in assisted_ring_calibration and not isinstance(
        assisted_ring_calibration.get("enabled"), bool
    ):
        errors.append("parameter_sweep.assisted_ring_calibration.enabled must be boolean")

    for name, vals in (("probe_delay_values", probe_delay_values),):
        if vals is None:
            errors.append(f"parameter_sweep.{name} is missing for probe5 mode")
        elif not vals:
            errors.append(f"parameter_sweep.{name} is empty for probe5 mode")
        elif not all(isinstance(v, (int, float)) and float(v) > 0 for v in vals):
            errors.append(f"parameter_sweep.{name} must all be positive numbers")

    if probe_energy_values is None:
        errors.append("parameter_sweep.probe_energy_values is missing for probe5 mode")
    elif not probe_energy_values:
        errors.append("parameter_sweep.probe_energy_values is empty for probe5 mode")
    elif ring_calibration_enabled:
        for value in probe_energy_values:
            if not _valid_power_expr(value):
                errors.append(
                    "parameter_sweep.probe_energy_values must be non-negative numbers "
                    "or P-relative expressions in calibrated probe5 mode"
                )
                break
            try:
                if resolve_power(value, 1.0) < 0:
                    errors.append("parameter_sweep.probe_energy_values must be non-negative")
                    break
            except (TypeError, ValueError):
                pass
        if any(
            isinstance(value, str) and _PROBE_POWER_REF_RE.match(value)
            for value in probe_energy_values
        ) and not probe_calibration_enabled:
            errors.append(
                "parameter_sweep.probe_energy_values uses P_probe, but "
                "parameter_sweep.probe_calibration.enabled is not true"
            )
        if any(
            isinstance(value, str) and _ASSISTED_PROBE_POWER_REF_RE.match(value)
            for value in probe_energy_values
        ) and not assisted_calibration_enabled:
            errors.append(
                "parameter_sweep.probe_energy_values uses P_assisted_probe, but "
                "parameter_sweep.assisted_probe_calibration.enabled is not true"
            )
        if any(
            isinstance(value, str) and _ASSISTED_RING_POWER_REF_RE.match(value)
            for value in probe_energy_values
        ):
            errors.append(
                "parameter_sweep.probe_energy_values must not use P_assisted_ring; "
                "P_assisted_ring is a ring-strength reference, not a center-probe reference"
            )
    elif not all(
        isinstance(v, (int, float)) and float(v) >= 0 for v in probe_energy_values
    ):
        errors.append("parameter_sweep.probe_energy_values must all be non-negative numbers")

    ring_axis_count = sum(
        axis is not None
        for axis in (ring_scale_values, ring_multiplier_values, ring_energy_values)
    )
    if ring_axis_count > 1:
        errors.append(
            "parameter_sweep.ring_scale_values, ring_multiplier_values and "
            "ring_energy_values are mutually exclusive"
        )
    if ring_scale_values is not None:
        if not ring_scale_values:
            errors.append("parameter_sweep.ring_scale_values is empty")
        elif not all(
            isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
            for v in ring_scale_values
        ):
            errors.append(
                "parameter_sweep.ring_scale_values must all be in the range [0, 1]"
            )
    if ring_multiplier_values is not None:
        if not ring_multiplier_values:
            errors.append("parameter_sweep.ring_multiplier_values is empty")
        elif not all(
            isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0
            for v in ring_multiplier_values
        ):
            errors.append(
                "parameter_sweep.ring_multiplier_values must all be positive finite numbers"
            )
        ring_ref = str(sweep_cfg.get("ring_multiplier_reference", "P_ring"))
        ring_ref_norm = ring_ref.lower().replace("_", "")
        if ring_ref_norm not in {"pring", "passistedring", "pringassisted"}:
            errors.append(
                "parameter_sweep.ring_multiplier_reference must be P_ring or "
                "P_assisted_ring"
            )
        if ring_ref_norm in {"passistedring", "pringassisted"} and not (
            assisted_ring_calibration_enabled and probe_calibration_enabled
        ):
            errors.append(
                "parameter_sweep.ring_multiplier_reference=P_assisted_ring requires "
                "both assisted_ring_calibration.enabled and probe_calibration.enabled"
            )

    for name in (("square_side",) if ring_calibration_enabled else ("square_side", "ring_energy")):
        val = sweep_cfg.get(name)
        if val is None:
            errors.append(f"parameter_sweep.{name} is missing for probe5 mode")
        elif not (isinstance(val, (int, float)) and float(val) > 0):
            errors.append(f"parameter_sweep.{name}={val!r} must be a positive number")

    if ring_calibration_enabled:
        safe_fraction = ring_calibration.get("safe_fraction")
        try:
            sf = float(safe_fraction)
            if not (0.0 < sf < 1.0):
                errors.append(
                    "parameter_sweep.ring_calibration.safe_fraction must be in (0, 1)"
                )
        except (TypeError, ValueError):
            errors.append(
                "parameter_sweep.ring_calibration.safe_fraction must be a number in (0, 1)"
            )

        if assisted_calibration_enabled:
            ring_fraction = assisted_calibration.get("ring_fraction", safe_fraction)
            try:
                rf = float(ring_fraction)
                if not (0.0 < rf < 1.0):
                    errors.append(
                        "parameter_sweep.assisted_probe_calibration.ring_fraction "
                        "must be in (0, 1)"
                    )
            except (TypeError, ValueError):
                errors.append(
                    "parameter_sweep.assisted_probe_calibration.ring_fraction "
                    "must be a number in (0, 1)"
                )

        if assisted_ring_calibration_enabled:
            probe_fraction = assisted_ring_calibration.get("probe_fraction")
            try:
                pf = float(probe_fraction)
                if not (0.0 < pf < 1.0):
                    errors.append(
                        "parameter_sweep.assisted_ring_calibration.probe_fraction "
                        "must be in (0, 1)"
                    )
            except (TypeError, ValueError):
                errors.append(
                    "parameter_sweep.assisted_ring_calibration.probe_fraction "
                    "must be a number in (0, 1)"
                )
            guard_fraction = assisted_ring_calibration.get("ring_guard_fraction", 0.95)
            try:
                gf = float(guard_fraction)
                if not (0.0 < gf <= 1.0):
                    errors.append(
                        "parameter_sweep.assisted_ring_calibration.ring_guard_fraction "
                        "must be in (0, 1]"
                    )
            except (TypeError, ValueError):
                errors.append(
                    "parameter_sweep.assisted_ring_calibration.ring_guard_fraction "
                    "must be a number in (0, 1]"
                )

        ts = cfg.get("global", {}).get("threshold_search", {})
        for key in ("power_values", "sigma_time_values", "pulse_separation_values"):
            values = ts.get(key)
            if not values:
                errors.append(f"threshold_search.{key} is required for calibrated probe5 mode")
        for key in ("max_runtime_minutes", "condensation_fraction"):
            if key not in ts:
                errors.append(f"threshold_search.{key} is required for calibrated probe5 mode")

    ring_energy_ref = sweep_cfg.get("ring_energy")
    if ring_energy_values is not None:
        if not ring_energy_values:
            errors.append("parameter_sweep.ring_energy_values is empty")
        elif not all(
            isinstance(v, (int, float)) and float(v) > 0
            for v in ring_energy_values
        ):
            errors.append("parameter_sweep.ring_energy_values must all be positive numbers")
        elif isinstance(ring_energy_ref, (int, float)) and any(
            float(v) > float(ring_energy_ref) for v in ring_energy_values
        ):
            errors.append(
                "parameter_sweep.ring_energy_values must not exceed ring_energy; "
                "treat ring_energy as the verified subthreshold cap"
            )

    base_scenario_filter = sweep_cfg.get("base_scenarios")
    if base_scenario_filter:
        selected = (
            {str(n) for n in base_scenario_filter}
            if isinstance(base_scenario_filter, list)
            else {str(base_scenario_filter)}
        )
        for sc in cfg.get("scenarios", []):
            if sc.get("name") in selected:
                laser_ids = [
                    str(ld.get("id", f"laser_{i}"))
                    for i, ld in enumerate(sc.get("lasers", []))
                ]
                if probe_laser_id not in laser_ids:
                    errors.append(
                        f"probe5 probe_laser_id='{probe_laser_id}' not found in scenario "
                        f"'{sc['name']}' (available: {laser_ids})"
                    )
                if len(laser_ids) < 5:
                    errors.append(
                        f"probe5 scenario '{sc['name']}' must have at least 5 lasers "
                        f"(4 ring + 1 probe), found {len(laser_ids)}"
                    )
                if probe_laser_id in laser_ids[:4]:
                    errors.append(
                        f"probe5 scenario '{sc['name']}' has probe_laser_id="
                        f"'{probe_laser_id}' inside the first four lasers. "
                        "The first four lasers must be the square ring because "
                        "square-side expansion moves lasers[:4]."
                    )

    cutoff_sigma = float(defaults.get("cutoff_sigma", 3.0))
    sigma_time = float(defaults.get("sigma_time", 1.5))
    pulse_sep = sweep_cfg.get("pulse_separation")
    solver = cfg.get("global", {}).get("solver", {})
    total_time = solver.get("total_time")

    if probe_delay_values and isinstance(total_time, (int, float)):
        max_delay = float(max(float(v) for v in probe_delay_values))
        probe_end = max_delay + 2.0 * cutoff_sigma * sigma_time
        if probe_end >= float(total_time):
            errors.append(
                f"probe5 max probe_delay={max_delay} + probe pulse support "
                f"2*{cutoff_sigma}*{sigma_time}={2*cutoff_sigma*sigma_time:.1f} ps = "
                f"{probe_end:.1f} ps >= global.solver.total_time={total_time}. "
                "Increase total_time."
            )

    if (
        probe_delay_values
        and pulse_sep is not None
        and isinstance(pulse_sep, (int, float))
    ):
        n_ring_pulses = 9
        for sc in cfg.get("scenarios", []):
            base_filter = sweep_cfg.get("base_scenarios", [])
            if sc.get("name") in (base_filter if isinstance(base_filter, list) else []):
                ring_laser = (sc.get("lasers") or [{}])[0]
                n_ring_pulses = int(ring_laser.get("n_pulses", 9))
                break
        pump_end = (n_ring_pulses - 1) * float(pulse_sep) + 2.0 * cutoff_sigma * sigma_time
        min_delay = float(min(float(v) for v in probe_delay_values))
        min_safe_delay = pump_end + 1.0
        if min_delay < min_safe_delay:
            errors.append(
                f"probe5 min probe_delay={min_delay} ps < estimated ring pump_end + 1 ps="
                f"{min_safe_delay:.1f} ps. Ring pump_end="
                f"{pump_end:.1f} ps (({n_ring_pulses}-1)*{pulse_sep}+2*{cutoff_sigma}*{sigma_time}). "
                "Probe must start after the ring train has fully ended."
            )

    return errors


def _probe5_threshold_criterion(run_dir: str) -> float:
    path = os.path.join(run_dir, "threshold_result.json")
    if not os.path.isfile(path):
        return _PROBE5_THRESHOLD_CRITERION
    try:
        with open(path) as f:
            threshold = json.load(f)
        return float(
            (threshold.get("calibration") or {}).get(
                "condensation_psi_sq_threshold",
                _PROBE5_THRESHOLD_CRITERION,
            )
        )
    except (TypeError, ValueError, OSError, json.JSONDecodeError):
        return _PROBE5_THRESHOLD_CRITERION


def _post_probe_max(
    data: Any,
    time_arr: Any,
    key: str,
    probe_delay: float,
) -> float:
    if key not in data:
        return float("nan")
    arr = data[key]
    if time_arr is not None and len(time_arr) == len(arr):
        arr = arr[time_arr >= probe_delay]
    return float(arr.max()) if len(arr) > 0 else float("nan")


def _time_of_max(
    data: Any,
    time_arr: Any,
    key: str,
    probe_delay: float,
) -> float:
    if key not in data or time_arr is None:
        return float("nan")
    arr = data[key]
    if len(time_arr) != len(arr):
        return float("nan")
    mask = time_arr >= probe_delay
    arr_post = arr[mask]
    t_post = time_arr[mask]
    if len(arr_post) == 0:
        return float("nan")
    return float(t_post[arr_post.argmax()])


def _unique_float_values(rows: list[dict], key: str) -> list[float]:
    values: set[float] = set()
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fvalue):
            values.add(fvalue)
    return sorted(values)


def _compact_float_label(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _generate_probe5_heatmap(
    rows: list[dict],
    results_dir: str,
    metric: str,
    filename: str,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    delays = sorted({float(r["probe_delay_ps"]) for r in rows if r.get("probe_delay_ps") is not None})
    energies = sorted({float(r["probe_energy"]) for r in rows if r.get("probe_energy") is not None})
    if not delays or not energies:
        return

    z = np.full((len(energies), len(delays)), float("nan"))
    delay_idx = {d: i for i, d in enumerate(delays)}
    energy_idx = {e: i for i, e in enumerate(energies)}

    for r in rows:
        d = r.get("probe_delay_ps")
        e = r.get("probe_energy")
        v = r.get(metric)
        if d is None or e is None or v is None:
            continue
        if float(d) in delay_idx and float(e) in energy_idx and np.isfinite(float(v)):
            z[energy_idx[float(e)], delay_idx[float(d)]] = float(v)

    fig, ax = plt.subplots(figsize=(max(6, len(delays) * 0.55), max(4, len(energies) * 0.28)))
    im = ax.pcolormesh(delays, energies, z, shading="nearest", cmap="hot")
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel("probe delay (ps)")
    ax.set_ylabel("probe energy")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close(fig)


def _generate_probe5_ring_heatmap(
    rows: list[dict],
    results_dir: str,
    metric: str,
    filename: str,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x_key = "ring_scale"
    x_values = _unique_float_values(rows, x_key)
    x_label = "ring scale"
    if not x_values:
        x_key = "ring_energy"
        x_values = _unique_float_values(rows, x_key)
        x_label = "ring energy"

    probe_energies = _unique_float_values(rows, "probe_energy")
    if not x_values or not probe_energies:
        return

    z = np.full((len(probe_energies), len(x_values)), float("nan"))
    x_idx = {x: i for i, x in enumerate(x_values)}
    e_idx = {e: i for i, e in enumerate(probe_energies)}

    for row in rows:
        x = row.get(x_key)
        e = row.get("probe_energy")
        v = row.get(metric)
        if x is None or e is None or v is None:
            continue
        try:
            xf = float(x)
            ef = float(e)
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if xf in x_idx and ef in e_idx and np.isfinite(vf):
            z[e_idx[ef], x_idx[xf]] = vf

    fig, ax = plt.subplots(
        figsize=(max(6, len(x_values) * 0.55), max(4, len(probe_energies) * 0.45))
    )
    im = ax.pcolormesh(x_values, probe_energies, z, shading="nearest", cmap="hot")
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel(x_label)
    ax.set_ylabel("probe energy")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close(fig)


def _generate_probe5_selected_cases(rows: list[dict], results_dir: str) -> None:
    from pipeline.manifest.io import atomic_write_json

    above = [r for r in rows if r.get("crossed_threshold")]
    below = [
        r for r in rows
        if not r.get("crossed_threshold")
        and np.isfinite(float(r.get("central_roi_peak_psi_sq", float("nan"))))
    ]

    cases: dict = {}
    if rows:
        cases["max_response"] = max(
            rows, key=lambda r: float(r.get("central_roi_peak_psi_sq") or 0.0)
        )
        cases["min_response"] = min(
            rows, key=lambda r: float(r.get("central_roi_peak_psi_sq") or float("inf"))
        )
    if above:
        cases["first_above_threshold"] = min(
            above, key=lambda r: float(r.get("probe_energy") or float("inf"))
        )
    if below:
        cases["closest_below_threshold"] = max(
            below, key=lambda r: float(r.get("central_roi_peak_psi_sq") or 0.0)
        )

    probe_rows = [r for r in rows if float(r.get("probe_energy") or 0.0) > 0.0]
    ring_only_rows = [r for r in rows if float(r.get("probe_energy") or 0.0) == 0.0]
    above_probe = [r for r in probe_rows if r.get("crossed_threshold")]
    below_probe = [
        r for r in probe_rows
        if not r.get("crossed_threshold")
        and np.isfinite(float(r.get("central_roi_peak_psi_sq", float("nan"))))
    ]
    if above_probe:
        cases["lowest_ring_above_threshold"] = min(
            above_probe, key=lambda r: float(r.get("ring_energy") or float("inf"))
        )
    if below_probe:
        cases["highest_ring_below_threshold"] = max(
            below_probe, key=lambda r: float(r.get("ring_energy") or 0.0)
        )
    if ring_only_rows:
        cases["ring_only_max_response"] = max(
            ring_only_rows,
            key=lambda r: float(r.get("central_roi_peak_psi_sq") or 0.0),
        )

    atomic_write_json(
        os.path.join(results_dir, "selected_probe5_threshold_cases.json"), cases
    )


def _generate_probe5_summary(
    scenarios: list[str], run_dir: str, results_dir: str
) -> None:
    from pipeline.manifest.io import atomic_write_json, load_scenario_meta

    threshold_criterion = _probe5_threshold_criterion(run_dir)
    rows: list[dict] = []
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        sweep = meta.get("sweep") or {}
        if sweep.get("mode") != "probe5":
            continue

        sidecar_path = os.path.join(run_dir, f"{name}_scalars.npz")
        if not os.path.isfile(sidecar_path):
            continue

        data = np.load(sidecar_path)
        probe_delay = float(sweep.get("probe_delay_ps", 0.0))
        time_arr = data.get("time", None)
        prefix = "roi_center_D_circle"

        row: dict = {
            "scenario": name,
            "probe_delay_ps": probe_delay,
            "probe_energy": float(sweep.get("probe_energy", float("nan"))),
            "ring_energy": float(sweep.get("ring_energy", float("nan"))),
            "ring_scale": float(sweep.get("ring_scale", float("nan"))),
            "square_side_um": float(sweep.get("square_side", float("nan"))),
            "sigma_space": float(sweep.get("sigma_space", float("nan"))),
            "pump_end_ps": float(sweep.get("pump_end_ps", float("nan"))),
            "central_roi_peak_psi_sq": _post_probe_max(
                data, time_arr, f"{prefix}_mean_psi_sq", probe_delay
            ),
            "central_roi_peak_emission": _post_probe_max(
                data, time_arr, f"{prefix}_integral_emission", probe_delay
            ),
            "central_roi_peak_nR": _post_probe_max(
                data, time_arr, f"{prefix}_mean_nR", probe_delay
            ),
            "t_peak_emission_ps": _time_of_max(
                data, time_arr, f"{prefix}_integral_emission", probe_delay
            ),
        }
        row["crossed_threshold"] = (
            np.isfinite(row["central_roi_peak_psi_sq"])
            and row["central_roi_peak_psi_sq"] > threshold_criterion
        )
        rows.append(row)

    if not rows:
        return

    csv_path = os.path.join(results_dir, "probe5_threshold_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PROBE5_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(results_dir, "probe5_threshold_summary.json")
    atomic_write_json(json_path, rows)

    try:
        ring_values = _unique_float_values(rows, "ring_energy")
        delay_values = _unique_float_values(rows, "probe_delay_ps")
        if len(ring_values) > 1:
            for delay in delay_values:
                delay_rows = [
                    r for r in rows
                    if abs(float(r.get("probe_delay_ps", float("nan"))) - delay) < 1e-9
                ]
                suffix = (
                    f"_d{_compact_float_label(delay)}"
                    if len(delay_values) > 1
                    else ""
                )
                _generate_probe5_ring_heatmap(
                    delay_rows,
                    results_dir,
                    "central_roi_peak_psi_sq",
                    f"probe5_ring_sweep_heatmap_psi_sq{suffix}.png",
                    (
                        "Probe5 central |ψ|² peak vs ring strength"
                        f"  (threshold={threshold_criterion})"
                    ),
                )
                _generate_probe5_ring_heatmap(
                    delay_rows,
                    results_dir,
                    "central_roi_peak_emission",
                    f"probe5_ring_sweep_heatmap_emission{suffix}.png",
                    "Probe5 central emission peak vs ring strength",
                )
        else:
            _generate_probe5_heatmap(
                rows,
                results_dir,
                "central_roi_peak_psi_sq",
                "probe5_threshold_heatmap_psi_sq.png",
                f"Probe5 central |ψ|² peak  (threshold={threshold_criterion})",
            )
            _generate_probe5_heatmap(
                rows,
                results_dir,
                "central_roi_peak_emission",
                "probe5_threshold_heatmap_emission.png",
                "Probe5 central emission peak",
            )
    except Exception as e:
        import sys
        print(f"  WARNING: probe5 heatmap generation failed: {e}", file=sys.stderr)

    try:
        _generate_probe5_selected_cases(rows, results_dir)
    except Exception as e:
        import sys
        print(f"  WARNING: probe5 selected cases failed: {e}", file=sys.stderr)

    print(f"  Probe5 summary: {csv_path}  ({len(rows)} scenarios)")


class Probe5TrapGate:
    """4-ring-spot trap-gate experiment with a swept central probe."""

    name = "probe5_trap_gate"

    def matches(self, cfg: dict[str, Any]) -> bool:
        return probe5_sweep_enabled(cfg)

    def validate(self, cfg: dict[str, Any]) -> list[str]:
        g = cfg.get("global", {})
        return _validate_probe5_sweep(
            g.get("parameter_sweep", {}),
            g.get("laser_defaults", {}),
            cfg,
        )

    def expand_parameter_sweep(
        self, cfg: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
        return _expand_probe5_sweep(cfg)

    def build_calibration_scenarios(
        self, cfg: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None]:
        return build_probe5_calibration_scenarios(cfg)

    def summarize(
        self, scenarios: list[str], run_dir: str, results_dir: str
    ) -> bool:
        if not scenarios:
            return False
        _generate_probe5_summary(scenarios, run_dir, results_dir)
        return True
