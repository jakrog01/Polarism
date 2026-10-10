"""Reservoir registry helpers."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.reservoir.abstract_reservoir import AbstractReservoir

available_reservoirs: dict[str, type[AbstractReservoir]] = {}


def register_reservoir(name: str):
    """Register reservoir."""
    def decorator(cls: type[AbstractReservoir]) -> type[AbstractReservoir]:
        """Register the decorated object and return it."""
        existing = available_reservoirs.get(name)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"Reservoir '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_reservoirs[name] = cls
        return cls

    return decorator
