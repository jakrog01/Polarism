"""Config loading and sweep parameter extraction."""
from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str) -> dict[str, Any]:
    """Load and return the raw config dict from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_dataclass(cls: type[T], overrides: dict[str, Any]) -> T:
    """Construct *cls* from *overrides*, silently ignoring unknown keys."""
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in valid})


def get_sweep_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``sweep`` section of the config."""
    return cfg.get("sweep", {})


def get_laser_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``laser`` section of the config."""
    return cfg.get("laser", {})


def generate_power_values(sweep_cfg: dict[str, Any]) -> list[float]:
    """Generate the ordered list of power values for the sweep.

    Parameters
    ----------
    sweep_cfg : dict
        The ``sweep`` section of config.yaml.  Must contain ``P_min``,
        ``P_max``, and ``P_step``.

    Returns
    -------
    list[float]
        Power values from P_min to P_max (inclusive) in steps of P_step,
        rounded to 10 decimal places to avoid floating-point accumulation.
    """
    p_min = float(sweep_cfg["P_min"])
    p_max = float(sweep_cfg["P_max"])
    p_step = float(sweep_cfg["P_step"])
    n = int(round((p_max - p_min) / p_step)) + 1
    values = [round(p_min + i * p_step, 10) for i in range(n)]
    return [v for v in values if v <= p_max + 1e-9]
