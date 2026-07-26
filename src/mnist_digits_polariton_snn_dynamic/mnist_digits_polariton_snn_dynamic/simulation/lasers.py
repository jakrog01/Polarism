"""Construction of pulsed Gaussian pump arrays."""
from __future__ import annotations

import numpy as np

from polarism.config.simulation_parameters import LaserParameters
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.laser.pulse_gaussian import PulseGaussian

from mnist_digits_polariton_snn_dynamic.config.loader import PulseConfig


def build_lasers(
    positions: np.ndarray,
    powers: np.ndarray,
    sigma_space_um: float,
    pulse: PulseConfig,
    grid: SimulationGrid2D,
    precision: str = "double",
) -> list[PulseGaussian]:
    """Build pulsed Gaussian laser instances for arbitrary pump positions."""
    pos = np.asarray(positions, dtype=np.float64)
    p = np.asarray(powers, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions shape must be (n_spots, 2), got {pos.shape}")
    if p.shape != (pos.shape[0],):
        raise ValueError(f"powers shape must be ({pos.shape[0]},), got {p.shape}")
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
        for i in range(pos.shape[0])
    ]
