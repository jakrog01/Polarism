"""YAML config loader for mnist_spacetime_v1."""
from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_architecture_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("architecture", {})


def get_output_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("output", {})


def get_slurm_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("slurm", {})

