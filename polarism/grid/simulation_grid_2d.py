from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class SimulationGrid2D(Protocol):
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float
    dy: float
    X: Union["np.ndarray", "cp.ndarray"]
    Y: Union["np.ndarray", "cp.ndarray"]
    kx: Union["np.ndarray", "cp.ndarray"]
    ky: Union["np.ndarray", "cp.ndarray"]
    KX: Union["np.ndarray", "cp.ndarray"]
    KY: Union["np.ndarray", "cp.ndarray"]
    k_squared: Union["np.ndarray", "cp.ndarray"]
