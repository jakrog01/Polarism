from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import LaserParameters

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class AbstractLaser(ABC):
    P0: float
    x0: float
    y0: float
    P: Union[np.ndarray, cp.ndarray]

    def __init__(self, laser_config: LaserParameters, X: np.ndarray, Y: np.ndarray):
        self.xp = compute_engine.xp
        self.P0 = laser_config.P0
        self.x0 = laser_config.x0
        self.y0 = laser_config.y0
        self.delay = laser_config.delay
        self.P = self.xp.zeros_like(X)

    @abstractmethod
    def _amplitude(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _P_space(
        self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _P_time(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        raise NotImplementedError

    def get_power(
        self,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        t: float,
    ) -> Union[np.ndarray, cp.ndarray]:
        if t < self.delay:
            self.P = self.xp.zeros_like(X)
            return self.P
        t_eff = t - self.delay
        self.P = self._amplitude(t_eff) * self._P_space(X, Y) * self._P_time(t_eff)
        return self.P
