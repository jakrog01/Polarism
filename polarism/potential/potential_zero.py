from __future__ import annotations

import cupy as cp
import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registy import register_potential


@register_potential("zero")
def potential_zero(
    X: "np.ndarray | cp.ndarray",
    Y: "np.ndarray | cp.ndarray",
    cfg: PotentialParameters,
) -> "np.ndarray | cp.ndarray":
    return compute_engine.xp.zeros_like(X)
