"""Zero-potential helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registy import register_potential

if TYPE_CHECKING:
    import cupy as cp


@register_potential("zero")
def potential_zero(
    X: Union["np.ndarray", "cp.ndarray"],
    Y: Union["np.ndarray", "cp.ndarray"],
    cfg: PotentialParameters,
) -> Union["np.ndarray", "cp.ndarray"]:
    """Return a zero potential on the grid."""
    return compute_engine.xp.zeros_like(X)
