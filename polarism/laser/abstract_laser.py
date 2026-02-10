from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from polarism.config.simulation_parameters import LaserParameters


class AbstractLaser(ABC):
    P0: float
    x0: float
    y0: float
    P: np.ndarray

    def __init__(self, laser_config: LaserParameters, X: np.ndarray, Y: np.ndarray):
        self.P0 = laser_config.P0
        self.x0 = laser_config.x0
        self.y0 = laser_config.y0
        self.P = np.zeros_like(X)

    @abstractmethod
    def P_space(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def P_time(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def get_power(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        self.P = self.P_space(X, Y) * self.P_time(t)
        return self.P
