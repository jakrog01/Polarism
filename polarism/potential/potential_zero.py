from __future__ import annotations

import numpy as np

from polarism.potential.potential_registy import register_potential
from polarism.compute_engine import compute_engine

@register_potential("zero")
def potential_zero(X: np.ndarray) -> np.ndarray:
    return compute_engine.xp.zeros_like(X)
