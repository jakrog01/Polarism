from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import GridParameters


class ClosedIntervalSimulationGrid2D:
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

    def __init__(self, cfg: GridParameters):
        if cfg.nx < 2 or cfg.ny < 2:
            raise ValueError(
                "ClosedIntervalSimulationGrid2D requires nx >= 2 and ny >= 2"
            )

        xp = compute_engine.xp

        self.nx = cfg.nx
        self.ny = cfg.ny
        self.lx = cfg.lx
        self.ly = cfg.ly

        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)

        x = xp.linspace(-self.lx / 2, self.lx / 2, self.nx, dtype=xp.float64)
        y = xp.linspace(-self.ly / 2, self.ly / 2, self.ny, dtype=xp.float64)
        self.X, self.Y = xp.meshgrid(x, y, indexing="xy")

        self.kx = 2 * xp.pi * xp.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2 * xp.pi * xp.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = xp.meshgrid(self.kx, self.ky, indexing="xy")
        self.k_squared = self.KX**2 + self.KY**2
