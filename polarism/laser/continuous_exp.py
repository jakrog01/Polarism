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
    w: float
    cutoff_sigma: float

    def __init__(
        self,
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
    ):
        super().__init__(laser_config, X, Y)
        self.w = laser_config.sigma_space
        self.cutoff_sigma = laser_config.cutoff_sigma

    def _amplitude(self, t: float) -> float:
        return self.P0

    def _P_space(
        self,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
    ) -> Union[np.ndarray, cp.ndarray]:
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        r = self.xp.sqrt(r2)
        return self.xp.exp(-r / (self.w**2))

    def _P_time(self, t: float) -> float:
        return 1.0
