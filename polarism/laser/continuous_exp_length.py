"""Length-consistent continuous exponential laser model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.laser.continuous_exp import ContinuousExponentialPump
from polarism.laser.laser_registy import register_laser

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


@register_laser("continuous-exp-length")
class ContinuousExponentialLengthPump(ContinuousExponentialPump):
    """Represent a continuous exponential pump with a length decay scale."""

    def _P_space(
        self,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return ``exp(-r / sigma_space)`` with a 1/e decay length."""
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        return self.xp.exp(-self.xp.sqrt(r2) / self.w)
