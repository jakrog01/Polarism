"""Circular ROI masks and spatial integrals."""
from __future__ import annotations

from typing import Any

import numpy as np


def circular_roi_mask(
    grid_X: Any,
    grid_Y: Any,
    cx_um: float,
    cy_um: float,
    radius_um: float,
    xp: Any,
) -> Any:
    """Boolean mask True inside a circle of given radius centred at (cx, cy).

    Parameters
    ----------
    grid_X, grid_Y
        2D coordinate arrays (GPU or CPU).
    cx_um, cy_um
        Centre coordinates in μm.
    radius_um
        Circle radius in μm.
    xp
        Array module (numpy or cupy).

    Returns
    -------
    Bool array with the same shape as grid_X.
    """
    r2 = (grid_X - cx_um) ** 2 + (grid_Y - cy_um) ** 2
    return r2 <= radius_um ** 2


def roi_integral(field: Any, mask: Any, xp: Any) -> float:
    """Sum field values inside mask."""
    return float(xp.sum(field * mask))


def build_class_roi_masks(
    xs_um: np.ndarray,
    ys_um: np.ndarray,
    roi_radius_um: float,
    grid_X: Any,
    grid_Y: Any,
    xp: Any,
) -> list[Any]:
    """Build one circular mask per class spot.

    Parameters
    ----------
    xs_um, ys_um
        Spot centre coordinates, shape (n_classes,).
    roi_radius_um
        Radius of each ROI circle.
    grid_X, grid_Y
        2D coordinate arrays.
    xp
        Array module.

    Returns
    -------
    List of n_classes boolean masks.
    """
    return [
        circular_roi_mask(grid_X, grid_Y, float(xs_um[i]), float(ys_um[i]), roi_radius_um, xp)
        for i in range(len(xs_um))
    ]
