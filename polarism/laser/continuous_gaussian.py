"""Continuous Gaussian laser model."""
from __future__ import annotations
from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp

@register_laser("continuous-gaussian")
class ContinuousGaussian(AbstractLaser):
    """Represent a continuous gaussian."""
    sigma_space: float

    def __init__(self, laser_config: LaserParameters, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray], precision: str = "double"):
        """Set up the continuous gaussian."""
        super().__init__(laser_config, X, Y, precision)
        self.sigma_space = laser_config.sigma_space
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        """Return the base laser amplitude at time t."""
        return self.P0

    def _P_space(self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """Return the spatial pump profile."""
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        return self.xp.exp(-0.5 * r2 / self.sigma_space**2)

    def _P_time(self, t: float) -> float:
        """Return the time profile of the pump."""
        return 1.0
