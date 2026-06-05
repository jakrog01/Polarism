"""WTA class spot geometry: 10 spots uniformly distributed on a ring."""
from __future__ import annotations

import math

import numpy as np

N_CLASSES = 10


def class_spot_positions(
    radius_um: float,
    n_classes: int = N_CLASSES,
    rotation_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) positions of n_classes spots in μm.

    Parameters
    ----------
    radius_um
        Ring radius in μm.
    n_classes
        Number of class spots.  Default 10.
    rotation_rad
        Azimuthal offset of class-0 spot from +x axis.

    Returns
    -------
    xs, ys : arrays of shape (n_classes,)
    """
    angles = rotation_rad + np.arange(n_classes) * (2.0 * math.pi / n_classes)
    xs = radius_um * np.cos(angles)
    ys = radius_um * np.sin(angles)
    return xs, ys
