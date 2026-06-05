"""Config loading for mnist_wta_v1."""
from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_physics_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["global"]["physics"]


def get_grid_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["global"]["grid"]


def get_encoding_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["encoding"]


def get_readout_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["readout"]


def get_calibration_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["calibration"]


def get_mnist_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["mnist"]


def get_projection_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["projection"]


def get_classifier_cfg(cfg: dict) -> dict[str, Any]:
    return cfg.get("classifier", {"type": "ridge", "alpha": 1.0})


def get_amplitude_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["amplitude"]


def get_geometry_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["geometry"]


def get_slurm_cfg(cfg: dict) -> dict[str, Any]:
    return cfg["slurm"]
