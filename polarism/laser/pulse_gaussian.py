from __future__ import annotations

import numpy as np

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser


@register_laser("pulse-gaussian")
class PulseGaussian(AbstractLaser):
    sigma_space: float
    Pmax: float
    sigma_time: float
    sigma_space: float
    P0: float
    pulse_separation: float
    cutoff_sigma: float

    def __init__(self, laser_config: LaserParameters, X: np.ndarray, Y: np.ndarray):
        super().__init__(laser_config, X, Y)
        self.sigma_space = laser_config.sigma_space
        self.Pmax = laser_config.Pmax
        self.sigma_time = laser_config.sigma_time
        self.pulse_separation = laser_config.pulse_separation
        self.P0 = laser_config.P0
        self.cutoff_sigma = laser_config.cutoff_sigma

    def P_space(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        return np.exp(-0.5 * r2 / self.sigma_space**2)

    def P_time(self, t: float) -> float:
        n = round(t / self.pulse_separation)
        dt = t - n * self.pulse_separation

        if abs(dt) > self.cutoff_sigma * self.sigma_time:
            return 0.0

        return self.P0 * np.exp(-0.5 * (dt / self.sigma_time) ** 2)
