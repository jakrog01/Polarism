from __future__ import annotations

import os
import re
from dataclasses import fields as dc_fields
from typing import Any, TypeVar

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

T = TypeVar("T")


def load_config(path: str | None = None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_power(expr: str | float | None, p_threshold: float) -> float:
    if expr is None:
        return p_threshold
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, str):
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*P\s*$", expr)
        if m:
            return float(m.group(1)) * p_threshold
        return float(expr)
    raise ValueError(f"Invalid power expression: {expr!r}")


def make_dataclass(cls: type[T], overrides: dict[str, Any]) -> T:
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in valid})


def get_laser_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["global"].get("laser_defaults", {})


def get_threshold_search_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["global"]["threshold_search"]


def get_scenario(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for sc in cfg["scenarios"]:
        if sc["name"] == name:
            return sc
    raise KeyError(f"Scenario '{name}' not found in config")
