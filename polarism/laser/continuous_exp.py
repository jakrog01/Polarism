"""Continuous exponential laser model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


@register_laser("continuous-exp")
class ContinuousExponentialPump(AbstractLaser):
    """Represent a continuous exponential pump."""
    w: float
    cutoff_sigma: float

    def __init__(
        self,
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        precision: str = "double",
    ):
        """Set up the continuous exponential pump."""
        super().__init__(laser_config, X, Y, precision)
        self.w = laser_config.sigma_space
        self.cutoff_sigma = laser_config.cutoff_sigma
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t: float) -> float:
        """Return the base laser amplitude at time t."""
        return self.P0

    def _P_space(
        self,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the spatial pump profile."""
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        r = self.xp.sqrt(r2)
        return self.xp.exp(-r / (self.w**2))

    def _P_time(self, t: float) -> float:
        """Return the time profile of the pump."""
        return 1.0
