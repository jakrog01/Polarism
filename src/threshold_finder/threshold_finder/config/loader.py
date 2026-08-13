"""Config loading and sweep parameter extraction."""
from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str) -> dict[str, Any]:
    """Load and return the raw config dict from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_dataclass(cls: type[T], overrides: dict[str, Any]) -> T:
    """Construct *cls* from *overrides*, silently ignoring unknown keys."""
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in valid})


def get_sweep_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``sweep`` section of the config."""
    return cfg.get("sweep", {})


def get_laser_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``laser`` section of the config."""
    return cfg.get("laser", {})


def generate_power_values(sweep_cfg: dict[str, Any]) -> list[float]:
    """Generate the ordered list of power values for the sweep.

    Parameters
    ----------
    sweep_cfg : dict
        The ``sweep`` section of config.yaml.  Must contain ``P_min``,
        ``P_max``, and ``P_step``.

    Returns
    -------
    list[float]
        Power values from P_min to P_max (inclusive) in steps of P_step,
        rounded to 10 decimal places to avoid floating-point accumulation.
    """
    p_min = float(sweep_cfg["P_min"])
    p_max = float(sweep_cfg["P_max"])
    p_step = float(sweep_cfg["P_step"])
    n = int(round((p_max - p_min) / p_step)) + 1
    values = [round(p_min + i * p_step, 10) for i in range(n)]
    return [v for v in values if v <= p_max + 1e-9]


def generate_sweep_points(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate ordered scalar sweep points.

    The default sweep variable is pump power.  A separation sweep keeps pump
    power fixed and varies ``laser.pulse_separation``.
    """
    sweep_cfg = get_sweep_config(cfg)
    variable = str(sweep_cfg.get("variable", "P"))
    if variable in {"P", "power", "pulse_energy"}:
        points = [
            {"index": i, "P": p, "sweep_variable": "P", "sweep_value": p}
            for i, p in enumerate(generate_power_values(sweep_cfg))
        ]
        return _expand_seeds(points, sweep_cfg)
    if variable == "pulse_separation":
        p_fixed = float(sweep_cfg["P_fixed"])
        s_min = float(sweep_cfg["pulse_separation_min"])
        s_max = float(sweep_cfg["pulse_separation_max"])
        s_step = float(sweep_cfg["pulse_separation_step"])
        laser = get_laser_config(cfg)
        sigma_time = float(laser.get("sigma_time", 1.0))
        cutoff_sigma = float(laser.get("cutoff_sigma", 3.0))
        n_pulses = int(laser.get("n_pulses", 0))
        post_pulse_time = float(sweep_cfg.get("post_pulse_time", 80.0))
        adaptive_total_time = bool(sweep_cfg.get("adaptive_total_time", True))
        n = int(round((s_max - s_min) / s_step)) + 1
        separations = [
            round(s_min + i * s_step, 10)
            for i in range(n)
            if s_min + i * s_step <= s_max + 1e-9
        ]
        points = []
        for i, sep in enumerate(separations):
            point = {
                "index": i,
                "P": p_fixed,
                "pulse_separation": sep,
                "sweep_variable": "pulse_separation",
                "sweep_value": sep,
            }
            if adaptive_total_time and n_pulses > 0:
                pulse_support = 2.0 * cutoff_sigma * sigma_time
                point["total_time"] = round(pulse_support + (n_pulses - 1) * sep + post_pulse_time, 10)
            points.append(point)
        return _expand_seeds(points, sweep_cfg)
    raise ValueError(f"Unsupported sweep.variable={variable!r}")


def _expand_seeds(points: list[dict[str, Any]], sweep_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = sweep_cfg.get("seeds")
    if not seeds:
        return points
    return [
        {**point, "index": index, "seed": int(seed)}
        for index, (point, seed) in enumerate((point, seed) for point in points for seed in seeds)
    ]
