"""Expand compact spacetime mechanism configs into concrete scenarios."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def _safe_label(value: Any) -> str:
    text = str(value).strip().replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "x"


def _angles_to_positions(radius_um: float, angles_deg: list[float]) -> list[tuple[float, float]]:
    out = []
    for deg in angles_deg:
        theta = math.radians(float(deg))
        out.append((radius_um * math.cos(theta), radius_um * math.sin(theta)))
    return out


def _ring_feature_positions(rings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    idx = 0
    for ring_i, ring in enumerate(rings):
        n = int(ring["n_spots"])
        radius = float(ring["radius_um"])
        rotation = float(ring.get("rotation_deg", 0.0))
        for j in range(n):
            deg = rotation + 360.0 * j / n
            x, y = _angles_to_positions(radius, [deg])[0]
            features.append({
                "id": f"f{idx:02d}",
                "ring": ring_i,
                "angle_deg": deg,
                "x0": x,
                "y0": y,
            })
            idx += 1
    return features


def _roi_circle(roi_id: str, x0: float, y0: float, radius: float) -> dict[str, Any]:
    return {
        "id": roi_id,
        "shape": "circle",
        "x0": float(x0),
        "y0": float(y0),
        "radius": float(radius),
    }


def _output_rois(output_cfg: dict[str, Any], roi_radius_um: float) -> list[dict[str, Any]]:
    radius = float(output_cfg.get("radius_um", 38.0))
    angles = [float(v) for v in output_cfg.get("angles_deg", [])]
    rois = []
    for i, (x, y) in enumerate(_angles_to_positions(radius, angles)):
        rois.append(_roi_circle(f"out_{i:02d}", x, y, roi_radius_um))
    return rois


def _base_laser(
    laser_id: str,
    x0: float,
    y0: float,
    power: float,
    delay: float,
    role: str,
    sigma_space_um: float,
    sigma_time_ps: float,
    cutoff_sigma: float,
    power_definition: str,
) -> dict[str, Any]:
    return {
        "id": laser_id,
        "role": role,
        "x0": float(x0),
        "y0": float(y0),
        "power": float(power),
        "delay": float(delay),
        "sigma_space_um": float(sigma_space_um),
        "sigma_time_ps": float(sigma_time_ps),
        "cutoff_sigma": float(cutoff_sigma),
        "power_definition": power_definition,
        "n_pulses": 1,
    }


def _expand_dynamic_hologram(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    arch = cfg["architecture"]
    geom = arch["geometry"]
    timing = arch.get("timing", {})
    pump = arch.get("pump", {})
    readout = arch.get("readout_probe", {})

    sigma_space = float(pump.get("sigma_space_um", 1.2))
    sigma_time = float(pump.get("sigma_time_ps", 1.5))
    cutoff = float(pump.get("cutoff_sigma", 3.0))
    power_def = str(pump.get("power_definition", "pulse_energy"))
    roi_radius = float(arch.get("roi_radius_um", 3.0))

    features = _ring_feature_positions(geom["feature_rings"])
    n_features = len(features)
    feature_scales = [float(v) for v in pump.get("feature_energy_scale_values", [600.0])]
    read_energies = [float(v) for v in readout.get("energy_values", [2200.0])]
    read_delays = [float(v) for v in timing.get("read_delay_values_ps", [25.0])]
    read_variants = readout.get("variants", [{"id": "center", "x0": 0.0, "y0": 0.0}])

    scenarios: list[dict[str, Any]] = []
    for pattern in arch.get("patterns", []):
        weights = [float(v) for v in pattern["weights"]]
        if len(weights) != n_features:
            raise ValueError(
                f"Pattern {pattern.get('name', '<unnamed>')} has {len(weights)} "
                f"weights, expected {n_features} feature spots"
            )
        pattern_name = _safe_label(pattern["name"])
        feature_delay = float(pattern.get("feature_delay_ps", 0.0))

        for feature_scale in feature_scales:
            for read_energy in read_energies:
                for read_delay in read_delays:
                    for variant in read_variants:
                        variant_id = _safe_label(variant.get("id", "read"))
                        lasers = [
                            _base_laser(
                                f["id"],
                                f["x0"],
                                f["y0"],
                                feature_scale * max(0.0, min(1.0, weights[i])),
                                feature_delay,
                                "feature",
                                sigma_space,
                                sigma_time,
                                cutoff,
                                power_def,
                            )
                            for i, f in enumerate(features)
                        ]
                        lasers.append(
                            _base_laser(
                                f"read_{variant_id}",
                                float(variant.get("x0", 0.0)),
                                float(variant.get("y0", 0.0)),
                                read_energy,
                                read_delay,
                                "read_probe",
                                float(variant.get("sigma_space_um", sigma_space)),
                                float(variant.get("sigma_time_ps", sigma_time)),
                                float(variant.get("cutoff_sigma", cutoff)),
                                power_def,
                            )
                        )

                        rois = [_roi_circle("center", 0.0, 0.0, roi_radius)]
                        if bool(arch.get("include_feature_rois", True)):
                            rois.extend(
                                _roi_circle(f["id"], f["x0"], f["y0"], roi_radius)
                                for f in features
                            )
                        rois.extend(_output_rois(arch.get("output_rois", {}), roi_radius))

                        name = (
                            f"holo_{pattern_name}_F{_safe_label(feature_scale)}"
                            f"_R{_safe_label(read_energy)}_d{_safe_label(read_delay)}"
                            f"_{variant_id}"
                        )
                        scenarios.append({
                            "name": name,
                            "architecture": "dynamic_hologram",
                            "pattern": pattern_name,
                            "lasers": lasers,
                            "rois": rois,
                            "metadata": {
                                "feature_scale": feature_scale,
                                "read_energy": read_energy,
                                "read_delay_ps": read_delay,
                                "read_variant": variant_id,
                                "n_feature_spots": n_features,
                                "description": pattern.get("description", ""),
                            },
                        })
    return scenarios


def _expand_ballistic_correlator(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    arch = cfg["architecture"]
    geom = arch["geometry"]
    pump = arch.get("pump", {})
    gate = arch.get("mixer_gate", {})

    sigma_space = float(pump.get("sigma_space_um", 1.2))
    sigma_time = float(pump.get("sigma_time_ps", 1.5))
    cutoff = float(pump.get("cutoff_sigma", 3.0))
    power_def = str(pump.get("power_definition", "pulse_energy"))
    roi_radius = float(arch.get("roi_radius_um", 3.0))

    input_radius = float(geom.get("input_radius_um", 28.0))
    input_angles = [float(v) for v in geom["input_angles_deg"]]
    input_positions = _angles_to_positions(input_radius, input_angles)
    input_energies = [float(v) for v in pump.get("input_energy_values", [1800.0])]
    gate_enabled = bool(gate.get("enabled", False))
    gate_energies = [float(v) for v in gate.get("energy_values", [0.0])]
    if not gate_enabled:
        gate_energies = [0.0]

    scenarios: list[dict[str, Any]] = []
    for pattern in arch.get("patterns", []):
        pattern_name = _safe_label(pattern["name"])
        active = {int(v) for v in pattern.get("active_inputs", [])}
        delays = {int(k): float(v) for k, v in (pattern.get("delays_ps") or {}).items()}
        for input_energy in input_energies:
            for gate_energy in gate_energies:
                lasers = []
                for i, (x, y) in enumerate(input_positions):
                    lasers.append(
                        _base_laser(
                            f"in_{i}",
                            x,
                            y,
                            input_energy if i in active else 0.0,
                            delays.get(i, 0.0),
                            "input",
                            sigma_space,
                            sigma_time,
                            cutoff,
                            power_def,
                        )
                    )
                if gate_enabled:
                    lasers.append(
                        _base_laser(
                            "mixer_gate",
                            float(gate.get("x0", 0.0)),
                            float(gate.get("y0", 0.0)),
                            gate_energy,
                            float(gate.get("delay_ps", 18.0)),
                            "mixer_gate",
                            float(gate.get("sigma_space_um", sigma_space)),
                            float(gate.get("sigma_time_ps", sigma_time)),
                            float(gate.get("cutoff_sigma", cutoff)),
                            power_def,
                        )
                    )

                rois = [_roi_circle("mixer", 0.0, 0.0, roi_radius)]
                rois.extend(
                    _roi_circle(f"in_{i}", x, y, roi_radius)
                    for i, (x, y) in enumerate(input_positions)
                )
                rois.extend(_output_rois(arch.get("output_rois", {}), roi_radius))
                name = (
                    f"corr_{pattern_name}_Ein{_safe_label(input_energy)}"
                    f"_G{_safe_label(gate_energy)}"
                )
                scenarios.append({
                    "name": name,
                    "architecture": "ballistic_correlator",
                    "pattern": pattern_name,
                    "lasers": lasers,
                    "rois": rois,
                    "metadata": {
                        "input_energy": input_energy,
                        "gate_energy": gate_energy,
                        "active_inputs": sorted(active),
                        "description": pattern.get("description", ""),
                    },
                })
    return scenarios


def expand_scenarios(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand configured architecture into concrete simulation scenarios."""
    arch = cfg.get("architecture", {})
    kind = str(arch.get("kind", "dynamic_hologram"))
    if kind == "dynamic_hologram":
        scenarios = _expand_dynamic_hologram(cfg)
    elif kind == "ballistic_correlator":
        scenarios = _expand_ballistic_correlator(cfg)
    else:
        raise ValueError(
            f"Unknown architecture.kind={kind!r}; expected dynamic_hologram "
            "or ballistic_correlator"
        )
    if not scenarios:
        raise ValueError("Scenario expansion produced no scenarios")
    names = [s["name"] for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("Scenario expansion produced duplicate names")
    return scenarios

