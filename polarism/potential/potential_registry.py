"""Potential registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PotentialParameters
from polarism.registry import Registry

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

Array = Union["np.ndarray", "cp.ndarray"]
PotentialFactory = Callable[[Array, Array, PotentialParameters], Array]

available_potentials: Registry[PotentialFactory] = Registry("potential")


def register_potential(name: str) -> Callable[[PotentialFactory], PotentialFactory]:
    """Return a decorator that registers a potential under ``name``."""
    return available_potentials.register(name)
