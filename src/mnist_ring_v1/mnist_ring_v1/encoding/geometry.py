"""Ring geometry: 16 spots uniformly distributed on a circle."""
from __future__ import annotations

import math

import numpy as np


N_RING_SPOTS = 16


def ring_spot_positions(
    radius_um: float,
    rotation_rad: float = math.pi / 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) positions of 16 ring spots in μm.

    Parameters
    ----------
    radius_um
        Ring radius in μm.
    rotation_rad
        Azimuthal offset of spot 0 from +x axis.  Default pi/16 ≈ 11.25 deg
        reduces alignment with grid axes.

    Returns
    -------
    xs, ys : arrays of shape (16,)
    """
    angles = rotation_rad + np.arange(N_RING_SPOTS) * (2.0 * math.pi / N_RING_SPOTS)
    xs = radius_um * np.cos(angles)
    ys = radius_um * np.sin(angles)
    return xs, ys
