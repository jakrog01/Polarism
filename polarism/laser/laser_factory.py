"""Laser factory helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import yaml

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_registy import available_lasers

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp
    from polarism.laser.abstract_laser import AbstractLaser


class LaserFactory:
    """Build laser objects from config data."""
    @staticmethod
    def create_laser(
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        precision: str = "double",
    ) -> list[AbstractLaser]:
        """Create one or more lasers from the config."""
        if laser_config.mode == "multiple":
            if laser_config.config_file is None:
                raise ValueError(
                    "config_file must be provided for multiple laser mode."
                )

            return LaserFactory._create_multiple_lasers(laser_config, X, Y, precision)

        return LaserFactory._create_single_laser(laser_config, X, Y, precision)

    @staticmethod
    def _create_single_laser(
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        precision: str = "double",
    ) -> list[AbstractLaser]:
        """Create one laser from the config."""
        laser_type = laser_config.laser_type
        if laser_type not in available_lasers:
            raise ValueError(
                f"Unknown laser type: '{laser_type}'. "
                f"Available: {list(available_lasers.keys())}"
            )
        return [available_lasers[laser_type](laser_config, X, Y, precision)]

    @staticmethod
    def _create_multiple_lasers(
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        precision: str = "double",
    ) -> list[AbstractLaser]:
        """Create all lasers listed in the config file."""
        with open(laser_config.config_file, "r") as f:
            data = yaml.safe_load(f)

        lasers = []
        for i, item in enumerate(data.get("lasers", [])):
            laser_type = item.get("laser_type")
            if laser_type is None:
                raise ValueError(
                    f"Laser entry {i} in '{laser_config.config_file}' "
                    f"is missing 'laser_type'."
                )
            if laser_type not in available_lasers:
                raise ValueError(
                    f"Unknown laser type '{laser_type}' in entry {i} of "
                    f"'{laser_config.config_file}'. "
                    f"Available: {list(available_lasers.keys())}"
                )
            individual_config = LaserParameters(**{**vars(laser_config), **item})
            lasers.append(available_lasers[laser_type](individual_config, X, Y, precision))
        return lasers


def normalize_config(cfg: LaserParameters | dict) -> dict:
    """Return the laser config as a plain dict."""
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    raise TypeError("Unsupported laser config format")
