"""Laser construction for spacetime mechanism scenarios."""
from __future__ import annotations

from typing import Any

import polarism.laser  # noqa: F401 - populate laser registry
from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_registy import available_lasers


def build_lasers(laser_defs: list[dict[str, Any]], grid_X: Any, grid_Y: Any) -> list[Any]:
    """Build Polarism pulse-gaussian lasers from expanded scenario definitions."""
    laser_cls = available_lasers.get("pulse-gaussian")
    if laser_cls is None:
        raise RuntimeError("pulse-gaussian laser not found in Polarism laser registry")

    lasers = []
    for item in laser_defs:
        power = float(item.get("power", 0.0))
        cfg = LaserParameters(
            mode="single",
            laser_type="pulse-gaussian",
            P0=power,
            Pmax=power,
            x0=float(item["x0"]),
            y0=float(item["y0"]),
            sigma_space=float(item.get("sigma_space_um", 1.2)),
            sigma_time=float(item.get("sigma_time_ps", 1.5)),
            pulse_separation=float(item.get("pulse_separation_ps", 500.0)),
            cutoff_sigma=float(item.get("cutoff_sigma", 3.0)),
            delay=float(item.get("delay", 0.0)),
            n_pulses=int(item.get("n_pulses", 1)),
            power_definition=str(item.get("power_definition", "pulse_energy")),
        )
        lasers.append(laser_cls(cfg, grid_X, grid_Y))
    return lasers
