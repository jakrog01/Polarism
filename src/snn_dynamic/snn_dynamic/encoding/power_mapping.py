"""Linear pixel-to-power encoding for the 49-spot pump lattice."""
from __future__ import annotations

import numpy as np


def encode_powers(
    images_7x7: np.ndarray,
    power_min: float,
    power_max: float,
) -> np.ndarray:
    """Map 7x7 image intensities linearly to 49 pump powers.

    Parameters
    ----------
    images_7x7
        Float64 images with shape ``(N, 7, 7)`` and values in ``[0, 1]``.
    power_min
        Pump value assigned to image value 0. Under ``pulse_energy`` this is
        per-pulse energy in micrometer squared per picosecond units.
    power_max
        Pump value assigned to image value 1. Under ``pulse_energy`` this is
        per-pulse energy in micrometer squared per picosecond units.

    Returns
    -------
    np.ndarray
        Float64 powers with shape ``(N, 49)`` in row-major order matching
        :func:`snn_dynamic.simulation.lasers_49.build_49_positions`.
    """
    images = np.asarray(images_7x7, dtype=np.float64)
    if images.ndim != 3 or images.shape[1:] != (7, 7):
        raise ValueError(f"Expected images with shape (N, 7, 7), got {images.shape}")
    if not np.isfinite(power_min) or not np.isfinite(power_max):
        raise ValueError("Power bounds must be finite")
    if not power_min < power_max:
        raise ValueError("power_min must be smaller than power_max")
    if np.min(images) < 0.0 or np.max(images) > 1.0:
        raise ValueError("Image values must be in [0, 1]")
    powers = power_min + images.reshape(images.shape[0], 49) * (power_max - power_min)
    return np.ascontiguousarray(powers, dtype=np.float64)
