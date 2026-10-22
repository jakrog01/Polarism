"""Uniform laser model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registry import register_laser

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


@register_laser("uniform")
class UniformLaser(AbstractLaser):
    """Represent an uniform laser."""
    time_independent = True
    @classmethod
    def batch_power_coefficient(cls, lasers, times):
        times = np.asarray(times)[:, None]
        power = np.asarray([laser.P0 for laser in lasers])[None, :]
        delay = np.asarray([laser.delay for laser in lasers])[None, :]
        return np.broadcast_to(np.where(times < delay, 0.0, power),
                               (len(times), len(lasers))).copy()
    def __init__(
        self,
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        precision: str = "double",
    ):
        """Set up the uniform laser."""
        super().__init__(laser_config, X, Y, precision)
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t: float) -> float:
        """Return the base laser amplitude at time t."""
        return self.P0

    def _P_space(
        self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the spatial pump profile."""
        xp = compute_engine.xp
        return xp.ones_like(X, dtype=self._real_dtype)

    def _P_time(self, t: float) -> float:
        """Return the time profile of the pump."""
        return 1.0
