"""Build the 17-laser configuration for one MNIST image (16 ring + 1 trigger)."""
from __future__ import annotations

from typing import Any

import numpy as np

import polarism.laser  # noqa: F401 - populate registry
from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_registy import available_lasers


def build_ring_trigger_lasers(
    ring_xs: np.ndarray,
    ring_ys: np.ndarray,
    ring_delays: np.ndarray,
    ring_power: float,
    trigger_delay: float,
    trigger_power: float,
    sigma_space_um: float,
    sigma_time_ring_ps: float,
    sigma_time_trigger_ps: float,
    cutoff_sigma: float,
    power_definition: str,
    grid_X: Any,
    grid_Y: Any,
) -> list[Any]:
    """Construct 16 ring lasers + 1 central trigger laser.

    All lasers are pulse-gaussian, n_pulses=1 (TTFS).

    Parameters
    ----------
    ring_xs, ring_ys
        Spot positions in μm, shape (16,).
    ring_delays
        Delay for each ring spot in ps (Polarism `delay` field), shape (16,).
    ring_power
        Pulse energy for each ring spot.
    trigger_delay
        Delay for the central trigger in ps.
    trigger_power
        Pulse energy for the trigger.
    sigma_space_um
        Gaussian spatial width in μm (shared by ring and trigger).
    sigma_time_ring_ps
        Temporal Gaussian width for ring spots in ps.
    sigma_time_trigger_ps
        Temporal Gaussian width for the central trigger in ps.
    cutoff_sigma
        Pulse cutoff in units of sigma_time.
    power_definition
        'pulse_energy' or 'peak_amplitude'.
    grid_X, grid_Y
        Grid coordinate arrays from ``create_grid``.

    Returns
    -------
    list of 17 laser instances (ring[0..15], trigger).
    """
    laser_cls = available_lasers.get("pulse-gaussian")
    if laser_cls is None:
        raise RuntimeError("pulse-gaussian laser not found in registry")

    lasers = []
    for i in range(16):
        cfg = LaserParameters(
            mode="single",
            laser_type="pulse-gaussian",
            P0=ring_power,
            Pmax=ring_power,
            x0=float(ring_xs[i]),
            y0=float(ring_ys[i]),
            sigma_space=sigma_space_um,
            sigma_time=sigma_time_ring_ps,
            pulse_separation=200.0,
            cutoff_sigma=cutoff_sigma,
            delay=float(ring_delays[i]),
            n_pulses=1,
            power_definition=power_definition,
        )
        lasers.append(laser_cls(cfg, grid_X, grid_Y))

    trig_cfg = LaserParameters(
        mode="single",
        laser_type="pulse-gaussian",
        P0=trigger_power,
        Pmax=trigger_power,
        x0=0.0,
        y0=0.0,
        sigma_space=sigma_space_um,
        sigma_time=sigma_time_trigger_ps,
        pulse_separation=200.0,
        cutoff_sigma=cutoff_sigma,
        delay=trigger_delay,
        n_pulses=1,
        power_definition=power_definition,
    )
    lasers.append(laser_cls(trig_cfg, grid_X, grid_Y))

    return lasers
