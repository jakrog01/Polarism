from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


@register_laser("uniform")
class UniformLaser(AbstractLaser):
    def __init__(
        self,
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
    ):
        super().__init__(laser_config, X, Y)

    def _amplitude(self, t: float) -> float:
        return self.P0

    def _P_space(
        self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        xp = compute_engine.xp
        return xp.ones_like(X, dtype=xp.float64)

    def _P_time(self, t: float) -> float:
        return 1.0
