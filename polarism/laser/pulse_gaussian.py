"""Pulsed Gaussian laser model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp

@register_laser("pulse-gaussian")
class PulseGaussian(AbstractLaser):
    """Represent a pulse gaussian."""
    sigma_space: float
    Pmax: float
    sigma_time: float
    sigma_space: float
    P0: float
    pulse_separation: float
    cutoff_sigma: float
    n_pulses: int

    def __init__(self, laser_config: LaserParameters, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]):
        """Set up the pulse gaussian."""
        super().__init__(laser_config, X, Y)
        self.sigma_space = laser_config.sigma_space
        self.Pmax = laser_config.Pmax
        self.sigma_time = laser_config.sigma_time
        self.pulse_separation = laser_config.pulse_separation
        self.P0 = laser_config.P0
        self.cutoff_sigma = laser_config.cutoff_sigma
        self.n_pulses = laser_config.n_pulses
        self.phase = self.cutoff_sigma * self.sigma_time

    def _pulse_index(self, t: float) -> int:
        return max(0, round((t - self.phase) / self.pulse_separation))

    def _amplitude(self, t: float) -> float:
        """Return the base laser amplitude at time t."""
        n = self._pulse_index(t)
        if self.n_pulses > 0 and n >= self.n_pulses:
            return 0.0
        return min(self.P0 + n * (self.Pmax - self.P0), self.Pmax)

    def _P_space(self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """Return the spatial pump profile."""
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        return self.xp.exp(-0.5 * r2 / self.sigma_space**2)

    def _P_time(self, t: float) -> float:
        """Return the time profile of the pump."""
        n = self._pulse_index(t)
        if self.n_pulses > 0 and n >= self.n_pulses:
            return 0.0
        dt = t - n * self.pulse_separation - self.phase

        if abs(dt) > self.cutoff_sigma * self.sigma_time:
            return 0.0

        return self.xp.exp(-0.5 * (dt / self.sigma_time) ** 2)
