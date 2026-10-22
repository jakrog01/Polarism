"""Laser factory helpers."""
from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Union

import yaml

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_registry import available_lasers

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
        entry_overrides=None,
    ) -> list[AbstractLaser]:
        """Create one or more lasers from the config."""
        if laser_config.mode == "multiple":
            if laser_config.config_file is None:
                raise ValueError(
                    "config_file must be provided for multiple laser mode."
                )

            return LaserFactory._create_multiple_lasers(laser_config, X, Y, precision, entry_overrides)

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
        return [available_lasers.lookup(laser_type)(laser_config, X, Y, precision)]

    @staticmethod
    def _create_multiple_lasers(
        laser_config: LaserParameters,
        X: Union[np.ndarray, cp.ndarray],
        Y: Union[np.ndarray, cp.ndarray],
        precision: str = "double",
        entry_overrides=None,
    ) -> list[AbstractLaser]:
        """Create all lasers listed in the config file."""
        with open(laser_config.config_file, "r") as f:
            data = yaml.safe_load(f)

        lasers = []
        for i, item in enumerate(data.get("lasers", [])):
            item = {**item}
            for field, value in (entry_overrides or {}).get(i, {}).items():
                if field.startswith("extra."):
                    key = field[6:]
                    if key in item and key not in {f.name for f in fields(LaserParameters)}:
                        item[key] = value
                    else:
                        item["extra"] = {**item.get("extra", {}), key: value}
                else:
                    item[field] = value
            laser_type = item.get("laser_type")
            if laser_type is None:
                raise ValueError(
                    f"Laser entry {i} in '{laser_config.config_file}' "
                    f"is missing 'laser_type'."
                )
            try:
                laser_class = available_lasers.lookup(laser_type)
            except ValueError as exc:
                raise ValueError(
                    f"Laser entry {i} in '{laser_config.config_file}': {exc}"
                ) from exc
            known = {f.name for f in fields(LaserParameters)}
            extra = {
                **laser_config.extra,
                **item.get("extra", {}),
                **{k: v for k, v in item.items() if k not in known},
            }
            individual_config = LaserParameters(**{
                **vars(laser_config),
                **{k: v for k, v in item.items() if k in known},
                "extra": extra,
            })
            lasers.append(laser_class(individual_config, X, Y, precision))
        return lasers


def normalize_config(cfg: LaserParameters | dict) -> dict:
    """Return the laser config as a plain dict."""
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    raise TypeError("Unsupported laser config format")
