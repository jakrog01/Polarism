from __future__ import annotations

import numpy as np

from polarism.potential.potential_registy import register_potential


@register_potential("zero")
def potential_zero(X: np.ndarray) -> np.ndarray:
    return np.zeros_like(X)
