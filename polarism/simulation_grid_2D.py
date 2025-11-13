import numpy as np
from dataclasses import dataclass
from polarism.config.simulation_parameters import GridParameters

class SimulationGrid2D:
    def __init__(self, cfg: GridParameters):
        self.nx, self.ny = cfg.nx, cfg.ny
        self.lx, self.ly = cfg.lx, cfg.ly
        self.dx, self.dy = cfg.lx / cfg.nx, cfg.ly / cfg.ny
        
        x = (np.arange(self.nx) - self.nx // 2) * self.dx
        y = (np.arange(self.ny) - self.ny // 2) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.k_squared = self.KX**2 + self.KY**2
