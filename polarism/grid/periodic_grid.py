"""Periodic simulation grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.grid.grid_registry import register_grid

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import GridParameters


@register_grid("periodic")
class PeriodicGrid:
    """Store a periodic one- or two-dimensional simulation grid."""
    boundary = "periodic"
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float
    dy: float
    grid_type: str
    X: Union["np.ndarray", "cp.ndarray"]
    Y: Union["np.ndarray", "cp.ndarray"]
    kx: Union["np.ndarray", "cp.ndarray"]
    ky: Union["np.ndarray", "cp.ndarray"]
    KX: Union["np.ndarray", "cp.ndarray"]
    KY: Union["np.ndarray", "cp.ndarray"]
    k_squared: Union["np.ndarray", "cp.ndarray"]

    def __init__(self, cfg: GridParameters):
        """Set up coordinates and Fourier frequencies."""
        xp = compute_engine.xp

        self.ndim = cfg.ndim
        self.nx = cfg.nx
        self.lx = cfg.lx
        self.grid_type = "periodic"

        self.dx = self.lx / self.nx
        if self.ndim == 1:
            self.shape = (self.nx,)
            self.spacing = (self.dx,)
            self.lengths = (self.lx,)
            self.X = (xp.arange(self.nx, dtype=xp.float64) - (self.nx - 1) * 0.5) * self.dx
            self.kx = 2 * xp.pi * xp.fft.fftfreq(self.nx, d=self.dx)
            self.KX = self.kx
            self.k_squared = self.KX**2
            return

        self.ny = cfg.ny
        self.ly = cfg.ly
        self.dy = self.ly / self.ny
        self.shape = (self.ny, self.nx)
        self.spacing = (self.dy, self.dx)
        self.lengths = (self.ly, self.lx)

        x = (xp.arange(self.nx, dtype=xp.float64) - (self.nx - 1) * 0.5) * self.dx
        y = (xp.arange(self.ny, dtype=xp.float64) - (self.ny - 1) * 0.5) * self.dy
        self.X, self.Y = xp.meshgrid(x, y, indexing="xy")

        self.kx = 2 * xp.pi * xp.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2 * xp.pi * xp.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = xp.meshgrid(self.kx, self.ky, indexing="xy")
        self.k_squared = self.KX**2 + self.KY**2
