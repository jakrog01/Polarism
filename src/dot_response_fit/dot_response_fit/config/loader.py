"""Config loading for the dot-response-fit pipeline.

Reads the YAML config and exposes typed accessors for each section.
No validation and no dataclass construction happen here — those live in
:mod:`dot_response_fit.config.validator` and
:mod:`dot_response_fit.config.builder`.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str) -> dict[str, Any]:
    """Load and return the raw config dict from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def extract_mnist_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``mnist`` section of the config."""
    return cfg["mnist"]


def extract_encoding_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``encoding`` section of the config."""
    return cfg["encoding"]


def extract_reference_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``reference`` section of the config."""
    return cfg["reference"]


def extract_fit_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``fit`` section of the config.

    Parameters
    ----------
    cfg : dict
        Full parsed config dict.

    Returns
    -------
    dict
        Content of ``cfg["fit"]``.
    """
    return cfg["fit"]


def extract_time_response_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``time_response`` section (legacy; removed from pipeline).

    Raises
    ------
    KeyError
        If the section is absent, signalling the config has been migrated.
    """
    return cfg["time_response"]


def build_amplitude_array(amp_cfg: dict[str, Any]):
    """Build the amplitude sweep array from a config sub-dict.

    Parameters
    ----------
    amp_cfg : dict
        Must contain ``"start"``, ``"end"``, and ``"count"``.

    Returns
    -------
    np.ndarray
        Linearly spaced amplitude values.
    """
    import numpy as np

    return np.linspace(
        float(amp_cfg["start"]),
        float(amp_cfg["end"]),
        int(amp_cfg["count"]),
    )


def build_t_eval(tr_cfg: dict[str, Any]):
    """Build the time evaluation grid from a ``time_response`` sub-dict.

    Parameters
    ----------
    tr_cfg : dict
        Must contain ``"t_start"``, ``"t_end"``, and ``"n_points"``.

    Returns
    -------
    np.ndarray
        Linearly spaced time grid.
    """
    import numpy as np

    return np.linspace(
        float(tr_cfg["t_start"]),
        float(tr_cfg["t_end"]),
        int(tr_cfg["n_points"]),
    )


def make_dataclass(cls: type[T], overrides: dict[str, Any]) -> T:
    """Construct *cls* from *overrides*, ignoring unknown keys.

    Parameters
    ----------
    cls : type
        A dataclass type.
    overrides : dict
        Key-value pairs; unknown keys are silently ignored.

    Returns
    -------
    T
        Instance of *cls*.
    """
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in valid})


def get_scenario(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the scenario entry with the given *name*.

    Raises
    ------
    KeyError
        If no scenario with that name exists.
    """
    for sc in cfg.get("scenarios", []):
        if sc["name"] == name:
            return sc
    raise KeyError(f"Scenario '{name}' not found in config")
