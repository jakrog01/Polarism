"""Laser interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.config.simulation_parameters import LaserParameters

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class AbstractLaser(ABC):
    """Define the interface for laser profiles."""
    P0: float
    x0: float
    y0: float
    P: Union[np.ndarray, cp.ndarray]

    def __init__(
        self,
        laser_config: LaserParameters,
        X: np.ndarray,
        Y: np.ndarray,
        precision: str = "double",
    ):
        """Set up the shared laser state."""
        self.xp = compute_engine.xp
        self._real_dtype = real_dtype(self.xp, precision)
        self.P0 = laser_config.P0
        self.x0 = laser_config.x0
        self.y0 = laser_config.y0
        self.delay = laser_config.delay
        self.P = self.xp.zeros(X.shape, dtype=self._real_dtype)

    def _finalize_spatial_envelope(
        self,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
    ) -> None:
        """Precompute and cache the static spatial envelope.

        Must be called at the end of each concrete subclass __init__,
        after all subclass-specific parameters have been set.
        """
        self._spatial_envelope = self._P_space(X, Y).astype(self._real_dtype, copy=False)

    @abstractmethod
    def _amplitude(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        """Return the base laser amplitude at time t."""
        raise NotImplementedError

    @abstractmethod
    def _P_space(
        self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the spatial pump profile."""
        raise NotImplementedError

    @abstractmethod
    def _P_time(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        """Return the time profile of the pump."""
        raise NotImplementedError

    def get_power(
        self,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        t: float,
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the full pump field at time t."""
        if t < self.delay:
            self.P = self.xp.zeros(X.shape, dtype=self._real_dtype)
            return self.P
        t_eff = t - self.delay
        self.P = self._amplitude(t_eff) * self._spatial_envelope * self._P_time(t_eff)
        return self.P
