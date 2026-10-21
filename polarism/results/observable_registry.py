"""Result observable factory registry."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from polarism.results.result_node import ResultNode
    from polarism.simulation_controller import SimulationController

ObservableFactory = Callable[["SimulationController"], "ResultNode | list[ResultNode]"]
available_observables: dict[str, ObservableFactory] = {}


def register_observable(name: str):
    """Register a result observable factory."""
    def decorator(factory: ObservableFactory) -> ObservableFactory:
        """Register the decorated factory and return it."""
        existing = available_observables.get(name)
        if existing is not None and existing is not factory:
            warnings.warn(
                f"Observable '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_observables[name] = factory
        return factory

    return decorator
