"""Initial-condition registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import InitialConditionParameters
from polarism.registry import Registry

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.grid.simulation_grid import SimulationGrid

Array = Union["np.ndarray", "cp.ndarray"]
InitialConditionFactory = Callable[["SimulationGrid", InitialConditionParameters, object, object], Array]

available_initial_conditions: Registry[InitialConditionFactory] = Registry("initial condition")


def register_initial_condition(name: str) -> Callable[[InitialConditionFactory], InitialConditionFactory]:
    """Return a decorator that registers an initial condition under ``name``."""
    return available_initial_conditions.register(name)
