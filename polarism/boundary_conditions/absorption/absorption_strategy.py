"""Boundary absorption interfaces and helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
)

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class AbsorptionStrategy(ABC):
    """Define the interface for boundary absorption."""
    @abstractmethod
    def __init__(self):
        """Set up the absorption strategy."""
        pass

    @abstractmethod
    def get_potential_distribution(self) -> Union[np.ndarray, cp.ndarray]:
        """Return potential distribution."""
        pass

    @abstractmethod
    def apply_absorption(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply absorption."""
        return psi


def create_absorption_profile(
    ny: int, nx: int, cfg_absorption: BoundaryConditionParameters
) -> Union[np.ndarray, cp.ndarray]:
    """Create absorption profile."""
    xp = compute_engine.xp

    if cfg_absorption.mask_width_percent <= 0:
        return xp.zeros((ny, nx))

    width_x = max(1, int(nx * cfg_absorption.mask_width_percent))
    width_y = max(1, int(ny * cfg_absorption.mask_width_percent))

    width_x = min(int(nx / 2), width_x)
    width_y = min(int(ny / 2), width_y)

    x_ramp_base = xp.linspace(0, xp.pi / 2, width_x)
    y_ramp_base = xp.linspace(0, xp.pi / 2, width_y)

    if cfg_absorption.profile_type == "sin2":
        ramp_x_1d = xp.sin(x_ramp_base) ** 2
        ramp_y_1d = xp.sin(y_ramp_base) ** 2
    elif cfg_absorption.profile_type == "parabolic":
        ramp_x_1d = (x_ramp_base * (2 / xp.pi)) ** 2
        ramp_y_1d = (y_ramp_base * (2 / xp.pi)) ** 2
    else:
        raise ValueError("Unsupported profile type")

    profile_x = xp.zeros(nx)
    profile_y = xp.zeros(ny)

    profile_x[:width_x] = ramp_x_1d[::-1]
    profile_x[-width_x:] = ramp_x_1d

    profile_y[:width_y] = ramp_y_1d[::-1]
    profile_y[-width_y:] = ramp_y_1d

    profile_2d = xp.maximum(profile_y[:, xp.newaxis], profile_x[xp.newaxis, :])

    return profile_2d
