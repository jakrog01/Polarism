"""Potential registry helpers."""
from __future__ import annotations

from types import FunctionType

available_potentials = {}


def register_potential(name: str):
    """Register potential."""
    def decorator(func: FunctionType) -> FunctionType:
        """Register the decorated object and return it."""
        available_potentials[name] = func
        return func

    return decorator
