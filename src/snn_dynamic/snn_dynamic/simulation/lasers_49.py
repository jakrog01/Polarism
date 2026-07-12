"""Construction of the 49 pulsed Gaussian pump lattice."""
from __future__ import annotations

import numpy as np

from polarism.config.simulation_parameters import LaserParameters
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.laser.pulse_gaussian import PulseGaussian

from snn_dynamic.config.loader import PulseConfig


def build_49_positions(
    center_x_um: float,
    center_y_um: float,
    pitch_um: float,
) -> np.ndarray:
    """Build the 7x7 pump-center lattice.

    Parameters
    ----------
    center_x_um
        Lattice center x-coordinate in micrometers.
    center_y_um
        Lattice center y-coordinate in micrometers.
    pitch_um
        Nearest-neighbor lattice spacing in micrometers.

    Returns
    -------
    np.ndarray
        Float64 array with shape ``(49, 2)`` storing ``(x, y)`` centers.
    """
    offsets = (np.arange(7, dtype=np.float64) - 3.0) * np.float64(pitch_um)
    xs, ys = np.meshgrid(
        offsets + np.float64(center_x_um),
        offsets + np.float64(center_y_um),
        indexing="xy",
    )
    return np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64, copy=False)


def build_49_lasers(
    positions: np.ndarray,
    powers: np.ndarray,
    sigma_space_um: float,
    pulse: PulseConfig,
    grid: SimulationGrid2D,
    precision: str = "double",
) -> list[PulseGaussian]:
    """Build 49 pulsed Gaussian laser instances.

    Parameters
    ----------
    positions
        Float64 pump positions with shape ``(49, 2)``.
    powers
        Float64 pump powers with shape ``(49,)``.
    sigma_space_um
        Gaussian spatial width in micrometers.
    pulse
        Shared pulse-Gaussian temporal envelope settings.
    grid
        Simulation grid providing device coordinate arrays.
    precision
        Polarism laser precision flag.

    Returns
    -------
    list[PulseGaussian]
        Pulse-Gaussian lasers with invariant normalized spatial envelopes.
    """
    pos = np.asarray(positions, dtype=np.float64)
    p = np.asarray(powers, dtype=np.float64)
    if pos.shape != (49, 2):
        raise ValueError(f"positions shape must be (49, 2), got {pos.shape}")
    if p.shape != (49,):
        raise ValueError(f"powers shape must be (49,), got {p.shape}")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(p)):
        raise ValueError("positions and powers must be finite")
    return [
        PulseGaussian(
            LaserParameters(
                mode="single",
                laser_type="pulse-gaussian",
                P0=float(p[i]),
                Pmax=float(p[i]),
                x0=float(pos[i, 0]),
                y0=float(pos[i, 1]),
                sigma_space=float(sigma_space_um),
                sigma_time=float(pulse.sigma_time),
                pulse_separation=float(pulse.pulse_separation),
                n_pulses=int(pulse.n_pulses),
                cutoff_sigma=float(pulse.cutoff_sigma),
                power_definition=str(pulse.power_definition),
                expose_results=False,
            ),
            grid.X,
            grid.Y,
            precision=precision,
        )
        for i in range(49)
    ]
