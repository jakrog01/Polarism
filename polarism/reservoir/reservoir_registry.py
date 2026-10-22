"""Reservoir registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.reservoir.abstract_reservoir import AbstractReservoir

available_reservoirs: Registry[type[AbstractReservoir]] = Registry("reservoir")


def register_reservoir(name: str) -> Callable[[type[AbstractReservoir]], type[AbstractReservoir]]:
    """Return a decorator that registers a reservoir under ``name``."""
    return available_reservoirs.register(name)
