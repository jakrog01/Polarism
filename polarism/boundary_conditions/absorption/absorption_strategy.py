from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
)


class AbsorptionStrategy(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_potential_distribution(self) -> np.ndarray:
        pass

    @abstractmethod
    def apply_absorption(self, psi: np.ndarray) -> np.ndarray:
        return psi


def create_absorption_profile(
    shape, cfg_absorption: BoundaryConditionParameters
) -> np.ndarray:
    nx, ny = shape

    if cfg_absorption.mask_width_percent <= 0:
        return np.zeros(shape)

    width_x = max(1, int(nx * cfg_absorption.mask_width_percent))
    width_y = max(1, int(ny * cfg_absorption.mask_width_percent))

    width_x = min(int(nx / 2), width_x)
    width_y = min(int(ny / 2), width_y)

    x_ramp_base = np.linspace(0, np.pi / 2, width_x)
    y_ramp_base = np.linspace(0, np.pi / 2, width_y)

    if cfg_absorption.profile_type == "sin2":
        ramp_x_1d = np.sin(x_ramp_base) ** 2
        ramp_y_1d = np.sin(y_ramp_base) ** 2
    elif cfg_absorption.profile_type == "parabolic":
        ramp_x_1d = (x_ramp_base * (2 / np.pi)) ** 2
        ramp_y_1d = (y_ramp_base * (2 / np.pi)) ** 2
    else:
        raise ValueError("Unsupported profile type")

    profile_x = np.zeros(nx)
    profile_y = np.zeros(ny)

    profile_x[:width_x] = ramp_x_1d[::-1]
    profile_x[-width_x:] = ramp_x_1d

    profile_y[:width_y] = ramp_y_1d[::-1]
    profile_y[-width_y:] = ramp_y_1d

    profile_2d = np.maximum(profile_x[:, np.newaxis], profile_y[np.newaxis, :])

    return profile_2d
