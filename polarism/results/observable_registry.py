"""Result observable factory registry."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.results.result_node import ResultNode
    from polarism.simulation_controller import SimulationController

ObservableFactory = Callable[["SimulationController"], "ResultNode | list[ResultNode]"]

available_observables: Registry[ObservableFactory] = Registry("observable")


def register_observable(name: str) -> Callable[[ObservableFactory], ObservableFactory]:
    """Return a decorator that registers an observable factory under ``name``."""
    return available_observables.register(name)
