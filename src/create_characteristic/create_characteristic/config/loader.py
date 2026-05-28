"""Config loading and 2D grid parameter generation."""
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


def generate_grid_points(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate the flat list of 2D grid points for the characteristic map.

    Parameters
    ----------
    cfg : dict
        Full parsed config dict.

    Returns
    -------
    list[dict]
        Flat list ordered by (energy_index, sep_index).  Each entry contains:
        ``index``, ``energy_index``, ``sep_index``, ``pulse_energy``,
        ``pulse_separation``, and ``total_time`` (when adaptive_total_time=true).
    """
    sweep = get_sweep_config(cfg)
    laser = get_laser_config(cfg)

    e_min = float(sweep["energy_min"])
    e_max = float(sweep["energy_max"])
    e_step = float(sweep["energy_step"])
    s_min = float(sweep["separation_min"])
    s_max = float(sweep["separation_max"])
    s_step = float(sweep["separation_step"])

    sigma_time = float(laser.get("sigma_time", 1.0))
    cutoff_sigma = float(laser.get("cutoff_sigma", 3.0))
    n_pulses = int(laser.get("n_pulses", 0))
    post_pulse_time = float(sweep.get("post_pulse_time", 80.0))
    adaptive = bool(sweep.get("adaptive_total_time", True))

    ne = int(round((e_max - e_min) / e_step)) + 1
    ns = int(round((s_max - s_min) / s_step)) + 1

    energies = [
        round(e_min + i * e_step, 10)
        for i in range(ne)
        if e_min + i * e_step <= e_max + 1e-9
    ]
    separations = [
        round(s_min + i * s_step, 10)
        for i in range(ns)
        if s_min + i * s_step <= s_max + 1e-9
    ]

    points: list[dict[str, Any]] = []
    idx = 0
    for ei, energy in enumerate(energies):
        for si, sep in enumerate(separations):
            point: dict[str, Any] = {
                "index": idx,
                "energy_index": ei,
                "sep_index": si,
                "pulse_energy": energy,
                "pulse_separation": sep,
            }
            if adaptive and n_pulses > 0:
                pulse_support = 2.0 * cutoff_sigma * sigma_time
                point["total_time"] = round(
                    pulse_support + (n_pulses - 1) * sep + post_pulse_time, 10
                )
            points.append(point)
            idx += 1

    return points
