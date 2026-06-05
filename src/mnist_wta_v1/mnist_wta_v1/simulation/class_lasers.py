"""Build 10 pulsed class spot lasers for WTA amplitude encoding."""
from __future__ import annotations

from typing import Any

import numpy as np

import polarism.laser  # noqa: F401 — populate registry
from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_registy import available_lasers


def pulse_support_window_ps(
    delay_ps: float,
    sigma_time_ps: float,
    cutoff_sigma: float,
) -> tuple[float, float, float]:
    """Return (start, peak, end) for Polarism pulse-gaussian timing."""
    phase = cutoff_sigma * sigma_time_ps
    return delay_ps, delay_ps + phase, delay_ps + 2.0 * phase


def validate_min_inter_pulse_gap(
    delays_ps: np.ndarray,
    sigma_time_ps: float,
    cutoff_sigma: float,
    min_gap_ps: float,
) -> None:
    """Validate that sorted pulse support windows do not overlap."""
    if len(delays_ps) < 2:
        return
    order = np.argsort(delays_ps)
    windows = [
        (int(i), *pulse_support_window_ps(float(delays_ps[i]), sigma_time_ps, cutoff_sigma))
        for i in order
    ]
    for (prev_i, _prev_start, _prev_peak, prev_end), (next_i, next_start, _next_peak, _next_end) in zip(windows, windows[1:]):
        required_start = prev_end + min_gap_ps
        if next_start < required_start:
            raise ValueError(
                "Class pulse timing overlaps: "
                f"class {next_i} starts at {next_start:.3f} ps, "
                f"but class {prev_i} ends at {prev_end:.3f} ps "
                f"and min_gap={min_gap_ps:.3f} ps requires start >= {required_start:.3f} ps."
            )


def build_class_lasers(
    xs_um: np.ndarray,
    ys_um: np.ndarray,
    powers: np.ndarray,
    sigma_space_um: float,
    sigma_time_ps: float,
    cutoff_sigma: float,
    power_definition: str,
    grid_X: Any,
    grid_Y: Any,
    delays_ps: np.ndarray | None = None,
    pulse_timing_mode: str = "simultaneous",
    min_inter_pulse_gap_ps: float = 0.0,
) -> list[Any]:
    """Construct n_classes single-pulse class spot lasers.

    Default WTA uses simultaneous class pulses (all delay=0) with class-specific
    pulse energy.  Non-overlap validation is available for future custom timing
    variants, but is not applied to the simultaneous WTA baseline.

    Parameters
    ----------
    xs_um, ys_um
        Class spot positions, shape (n_classes,).
    powers
        Per-class pump powers, shape (n_classes,).
    sigma_space_um
        Gaussian spatial width in μm.
    sigma_time_ps
        Gaussian temporal width in ps.
    cutoff_sigma
        Temporal cutoff in units of sigma_time_ps.
    power_definition
        'pulse_energy' or 'peak_amplitude' (passed through to LaserParameters).
    grid_X, grid_Y
        Grid coordinate arrays.
    delays_ps
        Per-class Polarism delay values. If None, all delays are 0.
    pulse_timing_mode
        'simultaneous' for the default WTA baseline, or 'custom' to validate
        provided delays against min_inter_pulse_gap_ps.

    Returns
    -------
    list of n_classes pulse-gaussian laser instances.
    """
    laser_cls = available_lasers.get("pulse-gaussian")
    if laser_cls is None:
        raise RuntimeError("pulse-gaussian laser not found in registry")

    n_classes = len(xs_um)
    if delays_ps is None:
        delays = np.zeros(n_classes, dtype=np.float64)
    else:
        delays = np.asarray(delays_ps, dtype=np.float64)
        if delays.shape != (n_classes,):
            raise ValueError(f"delays_ps shape {delays.shape} != ({n_classes},)")

    if pulse_timing_mode not in {"simultaneous", "custom"}:
        raise ValueError(
            f"pulse_timing_mode={pulse_timing_mode!r} is unsupported; "
            "use 'simultaneous' or 'custom'."
        )
    if pulse_timing_mode == "custom":
        validate_min_inter_pulse_gap(
            delays,
            sigma_time_ps=sigma_time_ps,
            cutoff_sigma=cutoff_sigma,
            min_gap_ps=min_inter_pulse_gap_ps,
        )

    lasers = []
    for i in range(n_classes):
        p = float(powers[i])
        cfg = LaserParameters(
            mode="single",
            laser_type="pulse-gaussian",
            P0=p,
            Pmax=p,
            x0=float(xs_um[i]),
            y0=float(ys_um[i]),
            sigma_space=sigma_space_um,
            sigma_time=sigma_time_ps,
            pulse_separation=200.0,
            cutoff_sigma=cutoff_sigma,
            delay=float(delays[i]),
            n_pulses=1,
            power_definition=power_definition,
        )
        lasers.append(laser_cls(cfg, grid_X, grid_Y))

    return lasers
