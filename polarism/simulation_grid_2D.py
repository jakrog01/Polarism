from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import GridParameters
    import numpy as np
    import cupy as cp

class SimulationGrid2D:
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float
    dy: float
    X: Union[np.ndarray, cp.ndarray]
    Y: Union[np.ndarray, cp.ndarray]
    kx: Union[np.ndarray, cp.ndarray]
    ky: Union[np.ndarray, cp.ndarray]
    KX: Union[np.ndarray, cp.ndarray]
    KY: Union[np.ndarray, cp.ndarray]
    k_squared: Union[np.ndarray, cp.ndarray]

    def __init__(self, cfg: GridParameters):
        xp = compute_engine.xp
        self.nx, self.ny = cfg.nx, cfg.ny
        self.lx, self.ly = cfg.lx, cfg.ly
        self.dx, self.dy = cfg.lx / cfg.nx, cfg.ly / cfg.ny

        x = (xp.arange(self.nx) - self.nx // 2) * self.dx
        y = (xp.arange(self.ny) - self.ny // 2) * self.dy
        self.X, self.Y = xp.meshgrid(x, y, indexing="xy")

        self.kx = 2 * xp.pi * xp.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2 * xp.pi * xp.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = xp.meshgrid(self.kx, self.ky, indexing="xy")
        self.k_squared = self.KX**2 + self.KY**2
