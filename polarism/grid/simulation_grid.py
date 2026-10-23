"""Protocol for one- and two-dimensional simulation grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class SimulationGrid(Protocol):
    """Spatial grid interface shared by model components."""
    ndim: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    lengths: tuple[float, ...]
    nx: int
    lx: float
    dx: float
    grid_type: str
    boundary: str
    X: Union["np.ndarray", "cp.ndarray"]
    kx: Union["np.ndarray", "cp.ndarray"]
    KX: Union["np.ndarray", "cp.ndarray"]
    k_squared: Union["np.ndarray", "cp.ndarray"]
