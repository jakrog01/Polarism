"""Config loading and basic extraction utilities.

Responsible for: reading YAML, resolving power expressions, extracting
named sections from the config dict.  No validation, no dataclass
construction. Those live in validator.py and builder.py.
"""
from __future__ import annotations

import re
from dataclasses import fields as dc_fields
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str) -> dict[str, Any]:
    """Load and return the raw config dict from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_power(expr: str | float | None, p_threshold: float) -> float:
    """Resolve a power expression to a float.

    Supports numeric literals and fractional-P notation such as ``"0.6P"``.

    Parameters
    ----------
    expr : str, float, or None
        Power value or expression.  ``None`` returns *p_threshold* unchanged.
    p_threshold : float
        Reference threshold power.

    Returns
    -------
    float
        Resolved power in the same units as *p_threshold*.
    """
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
    """Construct *cls* from *overrides*, ignoring unknown keys.

    Parameters
    ----------
    cls : type
        A dataclass type.
    overrides : dict
        Key-value pairs; unknown keys (not fields of *cls*) are silently
        ignored so YAML sections can contain extra annotations.

    Returns
    -------
    T
        Instance of *cls*.
    """
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in valid})


def get_laser_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``laser_defaults`` dict from the ``global`` section."""
    return cfg["global"].get("laser_defaults", {})


def get_threshold_search_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``threshold_search`` sub-dict from the ``global`` section."""
    return cfg["global"]["threshold_search"]


def get_scenario(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the scenario entry with the given *name*.

    Raises
    ------
    KeyError
        If no scenario with that name exists.
    """
    for sc in cfg["scenarios"]:
        if sc["name"] == name:
            return sc
    raise KeyError(f"Scenario '{name}' not found in config")
