"""Potential registry helpers."""
from __future__ import annotations

import warnings
from types import FunctionType

available_potentials = {}


def register_potential(name: str):
    """Register potential."""
    def decorator(func: FunctionType) -> FunctionType:
        """Register the decorated object and return it."""
        existing = available_potentials.get(name)
        if existing is not None and existing is not func:
            warnings.warn(
                f"Potential '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_potentials[name] = func
        return func

    return decorator
