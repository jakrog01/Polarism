"""Double-well potential helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registy import register_potential

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


@register_potential("double-well-supergaussian")
def potential_double_well_supergaussian(
    X: "np.ndarray | cp.ndarray",
    Y: "np.ndarray | cp.ndarray",
    cfg: PotentialParameters,
) -> "np.ndarray | cp.ndarray":
    """Return the double-well super-Gaussian potential."""
    xp = compute_engine.xp

    def well(x0: float, y0: float, V0: float, w: float, order: float):
        """Return one super-Gaussian well."""
        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        return V0 * xp.exp(-((r2 / (w**2)) ** order))

    return well(cfg.x1, cfg.y1, cfg.V1, cfg.w1, cfg.order) + well(
        cfg.x2, cfg.y2, cfg.V2, cfg.w2, cfg.order
    )
